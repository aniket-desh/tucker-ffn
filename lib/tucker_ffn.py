"""tucker-core ffn and swiglu ffn building blocks (note section IV).

three modules used by exp10 and the lm training in lib/lm.py:

  TuckerFFN(d, r, s, diagonal_only)
      generalized ffn with two latent rank-r projections P, Q and an output
      basis R of rank s, joined by a learned core C in R^{s x r x r}:
          z_alpha = sum_{i,j} C_aij p_i SiLU(q_j),    p = P^T x,  q = Q^T x
          y = R z
      diagonal_only=True parameterizes C as a vector c in R^{min(r,s)}, so
      C[a,i,j] = c_i if a==i==j else 0. recovers a swiglu-shaped block at
      r=s and Cforced superdiagonal (note eq 11, recovery condition).

  SwiGLUFFN(d, m)
      standard swiglu block in unconstrained form:
          h_j(x) = (w_j^T x) * SiLU(g_j^T x)
          y = sum_j u_j h_j(x)
      with W, G in R^{d x m} (stored as (m,d) Linear weights to match HF
      convention, see lib/model_utils.get_swiglu_layers) and U in R^{m x d}.

  SwiGLUFFNAligned(d, m, P, Q, gate_assignment)
      aligned swiglu where w_l is restricted to span(P) via w_l = P a_l,
      a_l in R^r (learnable), and g_l is selected from columns of Q by a
      gate_assignment vector j_l in {0,...,r-1}^m (constructor arg, not
      learned — represents the matched-coordinates assumption of theorem 1).
      P, Q are registered as buffers (frozen), only U and {a_l} are learned.
      this is the comparison the separation theorem actually constrains.

all three use einsum for the bilinear contraction, kept small enough that
contraction order does not matter at the scales we test.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _he_init_(weight, fan_in):
    """he-uniform init given an explicit fan_in."""
    bound = math.sqrt(6.0 / fan_in)
    nn.init.uniform_(weight, -bound, bound)


class TuckerFFN(nn.Module):
    """tucker-core ffn (note section IV).

    parameters:
      P  (d, r)  — first latent projection
      Q  (d, r)  — gate / second latent projection
      R  (d, s)  — output projection (residual basis)
      C  (s, r, r) or c_diag (min(r,s),) when diagonal_only

    forward:
      p = x @ P              (b, r)
      q = x @ Q              (b, r)
      sq = silu(q)           (b, r)
      z = einsum("aij,bi,bj->ba", C, p, sq)
      y = z @ R^T

    diagonal_only=True forces c[a,i,j] = c_i * delta_{a,i,j}, recovering the
    swiglu superdiagonal constraint at width r=s.
    """

    def __init__(self, d, r, s=None, diagonal_only=False, bias=False,
                  diagonal_bias_init=False):
        """Tucker-core ffn.

        diagonal_bias_init: when True (and diagonal_only is False), initialize
            C with C[a,a,a] = 1/sqrt(r) on the superdiagonal and small i.i.d.
            noise off-diagonal. This warm-starts the layer near a swiglu-shaped
            block (note section IV recovery condition) so the model can deviate
            into off-diagonal interactions only when training prefers it.
            This is purely an initialization choice; the parameterization and
            forward computation are unchanged.
        """
        super().__init__()
        if s is None:
            s = r
        self.d = d
        self.r = r
        self.s = s
        self.diagonal_only = diagonal_only

        self.P = nn.Parameter(torch.empty(d, r))
        self.Q = nn.Parameter(torch.empty(d, r))
        self.R = nn.Parameter(torch.empty(d, s))
        _he_init_(self.P, d)
        _he_init_(self.Q, d)
        _he_init_(self.R, s)

        if diagonal_only:
            kdim = min(r, s)
            self.c_diag = nn.Parameter(torch.empty(kdim))
            nn.init.normal_(self.c_diag, mean=0.0, std=1.0 / math.sqrt(r))
            self.register_buffer("C", None, persistent=False)
        else:
            self.C = nn.Parameter(torch.empty(s, r, r))
            if diagonal_bias_init:
                nn.init.normal_(self.C, mean=0.0,
                                 std=0.1 / math.sqrt(r))
                kdim = min(r, s)
                with torch.no_grad():
                    idx = torch.arange(kdim)
                    self.C[idx, idx, idx] = 1.0 / math.sqrt(r)
            else:
                nn.init.normal_(self.C, mean=0.0, std=1.0 / math.sqrt(r))

        if bias:
            self.bias = nn.Parameter(torch.zeros(d))
        else:
            self.register_parameter("bias", None)

    def core(self):
        """return C in (s, r, r) form, materializing diagonal case."""
        if self.diagonal_only:
            kdim = min(self.r, self.s)
            C = torch.zeros(self.s, self.r, self.r,
                            device=self.c_diag.device, dtype=self.c_diag.dtype)
            idx = torch.arange(kdim, device=self.c_diag.device)
            C[idx, idx, idx] = self.c_diag
            return C
        return self.C

    def forward(self, x):
        p = x @ self.P                       # (..., r)
        q = x @ self.Q
        sq = F.silu(q)
        C = self.core()
        if self.diagonal_only:
            kdim = min(self.r, self.s)
            z = torch.zeros(*p.shape[:-1], self.s,
                            device=p.device, dtype=p.dtype)
            z[..., :kdim] = self.c_diag * p[..., :kdim] * sq[..., :kdim]
        else:
            z = torch.einsum("aij,...i,...j->...a", C, p, sq)
        y = z @ self.R.T
        if self.bias is not None:
            y = y + self.bias
        return y

    def num_params(self):
        n = self.P.numel() + self.Q.numel() + self.R.numel()
        if self.diagonal_only:
            n += self.c_diag.numel()
        else:
            n += self.C.numel()
        if self.bias is not None:
            n += self.bias.numel()
        return n


class SwiGLUFFN(nn.Module):
    """standard swiglu ffn, unconstrained.

    follows HF convention: weights stored as (m, d) for gate/up and (d, m)
    for down, so rows of gate_proj.weight are g_j^T. forward computes
        h_j = (w_j^T x) * silu(g_j^T x),   y = sum_j u_j h_j.
    """

    def __init__(self, d, m, bias=False):
        super().__init__()
        self.d = d
        self.m = m
        self.gate_proj = nn.Linear(d, m, bias=bias)
        self.up_proj = nn.Linear(d, m, bias=bias)
        self.down_proj = nn.Linear(m, d, bias=bias)
        # match LLaMA-style init: small std, fan-in dependent
        nn.init.normal_(self.gate_proj.weight, std=1.0 / math.sqrt(d))
        nn.init.normal_(self.up_proj.weight,   std=1.0 / math.sqrt(d))
        nn.init.normal_(self.down_proj.weight, std=1.0 / math.sqrt(m))

    def forward(self, x):
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


class SwiGLUFFNAligned(nn.Module):
    """aligned swiglu under matched-coordinates assumption.

    given fixed latent dictionaries P (d, r) and Q (d, r), each of m hidden
    units uses
        w_l = P a_l,            a_l in R^r (learned)
        g_l = Q[:, j_l],        j_l in {0,...,r-1} fixed (constructor arg)
    and the unit computes
        h_l(x) = (a_l^T (P^T x)) * silu(Q[:,j_l]^T x).
    output projection u_l in R^d is learned. P, Q are buffers (frozen).

    this is the swiglu hypothesis class that theorem 1 constrains: at width
    m, an aligned swiglu can express V_j = R C^(j) for each gate j only by
    choosing a_l vectors that span the column space of V_j, with the number
    of units assigned to gate j upper-bounded by rank(V_j).
    """

    def __init__(self, d, m, P, Q, gate_assignment):
        super().__init__()
        assert P.dim() == 2 and Q.dim() == 2 and P.shape == Q.shape
        assert P.shape[0] == d
        self.d = d
        self.m = m
        self.r = P.shape[1]
        self.register_buffer("P", P.detach().clone())
        self.register_buffer("Q", Q.detach().clone())
        # j_l in {0,...,r-1}, length m
        ga = torch.as_tensor(gate_assignment, dtype=torch.long)
        assert ga.numel() == m
        self.register_buffer("gate_assignment", ga)
        # a in R^{m, r}, u in R^{m, d}  (so y = sum_l u_l h_l = a-mat-projected)
        self.A = nn.Parameter(torch.empty(m, self.r))
        self.U = nn.Parameter(torch.empty(d, m))
        nn.init.normal_(self.A, std=1.0 / math.sqrt(self.r))
        nn.init.normal_(self.U, std=1.0 / math.sqrt(m))

    def forward(self, x):
        p = x @ self.P                        # (..., r)
        q = x @ self.Q                        # (..., r)
        gate_pre = q[..., self.gate_assignment]  # (..., m)
        # w_l^T x = a_l . p  =>  matmul p (..., r) by A^T (r, m)
        up_pre = p @ self.A.T                 # (..., m)
        h = up_pre * F.silu(gate_pre)
        return h @ self.U.T

    def num_params(self):
        return self.A.numel() + self.U.numel()


# ── helpers for building swiglu of given param budget ───────────────────────

def swiglu_params(d, m):
    """parameter count of unconstrained SwiGLUFFN(d, m), no bias."""
    return 3 * d * m


def tucker_params(d, r, s):
    """parameter count of TuckerFFN(d, r, s), no bias, full core."""
    return d * (2 * r + s) + s * r * r


def swiglu_width_for_params(d, target_params):
    """largest m such that 3*d*m <= target_params (rounded to nearest int)."""
    return max(1, int(round(target_params / (3 * d))))


# ── unit tests ──────────────────────────────────────────────────────────────

def _test_tucker_recovers_swiglu():
    """a TuckerFFN with diagonal core should equal a SwiGLU with matched
    weights, up to floating-point error."""
    torch.manual_seed(0)
    d, m = 16, 8
    tk = TuckerFFN(d, r=m, s=m, diagonal_only=True)

    # construct a swiglu whose weights are derived from tk's params:
    #   p_i = P[:, i], q_i = Q[:, i], r_a = R[:, a]
    #   tk forward: z_a = c_a * p_a * silu(q_a)  (since diagonal: a==i==j)
    #               y = sum_a R[:, a] z_a
    #   equivalently a swiglu with m units, where unit l has:
    #       w_l = c_l * P[:, l]   (so w_l^T x = c_l p_l)
    #       g_l = Q[:, l]
    #       u_l = R[:, l]
    sw = SwiGLUFFN(d, m, bias=False)
    with torch.no_grad():
        c = tk.c_diag
        # gate_proj.weight has rows g_l^T = Q[:, l]^T => weight = Q.T
        sw.gate_proj.weight.copy_(tk.Q.T)
        # up_proj.weight rows are w_l^T = (c_l * P[:, l])^T
        sw.up_proj.weight.copy_((tk.P * c.unsqueeze(0)).T)
        # down_proj.weight columns are u_l = R[:, l] => weight = R
        sw.down_proj.weight.copy_(tk.R)

    x = torch.randn(4, d)
    y_tk = tk(x)
    y_sw = sw(x)
    err = (y_tk - y_sw).abs().max().item()
    assert err < 1e-4, f"tucker-diagonal vs swiglu mismatch: {err}"
    return err


def _test_aligned_swiglu_matches_unconstrained():
    """a SwiGLUFFNAligned with full-rank P, Q dictionaries can express
    arbitrary unconstrained-swiglu w_l, g_l only when r >= d (P spans R^d
    and Q's columns are chosen). spot-check that aligned forward runs and
    is consistent with manual computation."""
    torch.manual_seed(1)
    d, r, m = 16, 16, 8
    P = torch.randn(d, r)
    Q = torch.randn(d, r)
    ga = torch.arange(m) % r
    sw = SwiGLUFFNAligned(d, m, P, Q, ga)

    x = torch.randn(3, d)
    y = sw(x)

    # manual: h_l = (a_l . (P^T x)) * silu(Q[:,j_l]^T x)
    p = x @ P
    q = x @ Q
    h_manual = (p @ sw.A.T) * F.silu(q[:, ga])
    y_manual = h_manual @ sw.U.T
    err = (y - y_manual).abs().max().item()
    assert err < 1e-5, f"aligned forward mismatch: {err}"
    return err


def _test_tucker_einsum_matches_loop():
    """tucker einsum contraction should match an explicit double loop."""
    torch.manual_seed(2)
    d, r, s = 8, 4, 5
    tk = TuckerFFN(d, r, s, diagonal_only=False)
    x = torch.randn(7, d)

    p = x @ tk.P
    q = x @ tk.Q
    sq = F.silu(q)
    C = tk.C

    z_loop = torch.zeros(7, s)
    for a in range(s):
        for i in range(r):
            for j in range(r):
                z_loop[:, a] += C[a, i, j] * p[:, i] * sq[:, j]
    y_loop = z_loop @ tk.R.T

    y_einsum = tk(x)
    err = (y_loop - y_einsum).abs().max().item()
    assert err < 1e-5, f"einsum vs loop mismatch: {err}"
    return err


if __name__ == "__main__":
    e1 = _test_tucker_recovers_swiglu()
    e2 = _test_aligned_swiglu_matches_unconstrained()
    e3 = _test_tucker_einsum_matches_loop()
    print(f"[done] tucker_recovers_swiglu  | err={e1:.2e}")
    print(f"[done] aligned_forward_match    | err={e2:.2e}")
    print(f"[done] tucker_einsum_vs_loop    | err={e3:.2e}")
