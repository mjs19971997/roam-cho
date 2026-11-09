"""Study rules of code/hl500 (PLAN_roamcho.md): ROAM on conditional handover (RCHO, MCHO and ablations) and ROAM
on Event A3 (RA3), written against the public HandoverLab rule API.  The library is not changed."""
import numpy as np

from handoverlab.rules.base import (BaselineRule, ConditionalRule, EARLY, LATE, UNNECESSARY, USEFUL,
                                    MAX_CONDITIONAL_CELLS)
from handoverlab.seeds import streams
from handoverlab.traffic import assign_services

# interruption weight w_s of the ROAM paper (roam-handover/roam/services.py)
W_SERVICE = {"voice": 0.35, "video_call": 0.55, "gaming": 0.70, "streaming": 0.20, "web": 0.10}

# 48-point sector quadrature of ROAM (6 equal-area rings x 8 angles)
_RR = np.sqrt((np.arange(6) + 0.5) / 6.0)
_AA = (np.arange(8) + 0.5) / 8.0 * 2 - 1
_QR, _QA = (g.ravel() for g in np.meshgrid(_RR, _AA))
_TAU = 0.25 * np.arange(1, 41)            # dwell predictor: 10 s horizon in 0.25 s steps


def service_columns(ctx):
    """Service index, T_min^s and w_s of every user: the world's own draw (streams(seed, 'world', 3)[1])."""
    sc = ctx.scenario
    svcs = sc.traffic.services
    rng_svc = streams(ctx.seed, "world", 3)[1]
    idx = assign_services(svcs, sc.mobility.users, rng_svc)
    tmin = np.array([s.t_min_s for s in svcs])[idx]
    wtab = np.array([W_SERVICE.get(s.key, np.nan) for s in svcs])
    share = np.array([s.share for s in svcs])
    wtab = np.where(np.isnan(wtab), np.nanmean(wtab), wtab)
    w_bar = float((wtab * share).sum() / share.sum())
    return idx, tmin, wtab[idx], w_bar


class _RoamParts:
    """HHT (route + location tables), path-area term, dwell predictor and service weights."""

    def _roam_reset(self, ctx):
        M = ctx.sites.shape[0]
        self.M = M
        self.ar = np.arange(ctx.n_users)
        self.svc_idx, self.tmin, self.w_s, self.w_bar = service_columns(ctx)
        # HHT
        self.N = np.zeros((M, M, 4))
        lo = ctx.sites.min(0) - 300.0
        hi = ctx.sites.max(0) + 300.0
        self.x0, self.y0 = lo
        self.nx = int((hi[0] - lo[0]) / self.d_b) + 1
        self.ny = int((hi[1] - lo[1]) / self.d_b) + 1
        self.NL = np.zeros((self.nx * self.ny, M, 4), dtype=np.float32)
        self.t_decay = 0.0
        # geometry state
        U = ctx.n_users
        self.f = np.zeros((U, M))
        self.s = np.zeros((U, M))
        self.t_geo = np.full(U, -np.inf)
        self.omega = np.zeros(U)
        self.th_prev = None
        self.t_prev = None

    def bins(self, x, y):
        ix = np.clip(((np.asarray(x) - self.x0) / self.d_b).astype(int), 0, self.nx - 1)
        iy = np.clip(((np.asarray(y) - self.y0) / self.d_b).astype(int), 0, self.ny - 1)
        return iy * self.nx + ix

    def _decay(self, t):
        acc = t - self.t_decay
        if acc >= 1.0 - 1e-9:
            fct = np.exp(-acc / self.tau_f)
            self.N *= fct
            self.NL *= fct
            self.t_decay = t

    def _scores(self, serv, b):
        """(H, L), each (U, M)."""
        if not self.use_memory:
            z = np.full((len(serv), self.M), 0.25)
            return z, z
        nr = self.N[serv] + self.beta
        pr = nr / nr.sum(2, keepdims=True)
        n = self.NL[b] + self.kappa_L * pr
        p = n / n.sum(2, keepdims=True)
        return self.alpha * p[..., 1] + (1 - self.alpha) * p[..., 2], p[..., 3]

    def _motion(self, obs):
        t, th = obs.t, np.asarray(obs.heading, dtype=float)
        if self.th_prev is not None and t > self.t_prev:
            dt = t - self.t_prev
            dth = np.abs((th - self.th_prev + np.pi) % (2 * np.pi) - np.pi)
            a = 0.9 ** (dt / 0.1)
            self.omega = a * self.omega + (1 - a) * dth / dt
        self.th_prev, self.t_prev = th.copy(), t
        return 1.0 / (1.0 + self.kappa * self.omega * self.T_p)

    def _geometry(self, obs):
        """Refresh f (area share) and s (short-stay indicator) of users older than 0.3 s."""
        u = np.nonzero(obs.t - self.t_geo >= 0.3 - 1e-9)[0]
        if len(u) == 0:
            return
        x, y = np.asarray(obs.x, float)[u], np.asarray(obs.y, float)[u]
        v, th = np.asarray(obs.speed, float)[u], np.asarray(obs.heading, float)[u]
        nc = self.ctx.nominal_cell
        if self.need_area:
            R = np.maximum(v * self.T_p, self.r0)[:, None] * _QR[None, :]
            ang = th[:, None] + self.phi * _QA[None, :]
            best = np.asarray(nc(x[:, None] + R * np.cos(ang), y[:, None] + R * np.sin(ang))).astype(int)
            f = np.zeros((len(u), self.M))
            np.add.at(f, (np.repeat(np.arange(len(u)), best.shape[1]), best.ravel()), 1.0 / best.shape[1])
            self.f[u] = f
        if self.need_dwell:
            d = v[:, None] * _TAU[None, :]
            cells = np.asarray(nc(x[:, None] + d * np.cos(th)[:, None], y[:, None] + d * np.sin(th)[:, None])).astype(int)
            K = cells.shape[1]
            inj = cells[:, :, None] == np.arange(self.M)[None, None, :]            # (u, K, M)
            anyin = inj.any(1)
            first = np.argmax(inj, axis=1)                                          # (u, M)
            kk = np.arange(K)[None, :, None]
            out_after = (~inj) & (kk > first[:, None, :])
            has_exit = out_after.any(1)
            exit_ = np.where(has_exit, np.argmax(out_after, axis=1), K)
            T_hat = np.where(anyin, (exit_ - first) * 0.25, 0.0)
            self.s[u] = np.clip(1.0 - T_hat / self.tmin[u][:, None], 0.0, 1.0)
        self.t_geo[u] = obs.t

    # -- learning --------------------------------------------------------
    def memory_key(self, obs):
        return self.bins(obs.x, obs.y)

    def on_outcome(self, rec):
        if rec.label not in (USEFUL, UNNECESSARY, EARLY, LATE) or rec.source < 0 or rec.target < 0:
            return
        self.N[rec.source, rec.target, rec.label] += 1.0
        if rec.key >= 0:
            self.NL[rec.key, rec.target, rec.label] += 1.0

    def on_link_failure(self, rec):
        if rec.source < 0 or rec.cell < 0 or rec.cell == rec.source:
            return
        self.N[rec.source, rec.cell, LATE] += 1.0
        self.NL[int(self.bins(rec.x, rec.y)), rec.cell, LATE] += 1.0


def _roam_defaults(obj, d_b=40.0, alpha=0.6, beta=1.0, kappa_L=3.0, tau_f=600.0, T_p=5.0, phi=0.6, r0=10.0, kappa=1.0):
    obj.d_b, obj.alpha, obj.beta, obj.kappa_L, obj.tau_f = d_b, alpha, beta, kappa_L, tau_f
    obj.T_p, obj.phi, obj.r0, obj.kappa = T_p, phi, r0, kappa


class RoamCHO(_RoamParts, ConditionalRule):
    """ROAM-CHO (PLAN_roamcho.md Section 1).  MCHO = eta_S = eta_D = 0.  memory=False: H = L = 1/4."""
    learns = True

    def __init__(self, o_prep=-2.0, o_exec=3.0, ttt_exec=0.08, n_max=4, eta_H=0.0, eta_L=0.0, eta_A=0.0,
                 eta_S=0.0, eta_D=0.0, rho=0.0, memory=True, name="RCHO"):
        super().__init__(name)
        if not 1 <= int(n_max) <= MAX_CONDITIONAL_CELLS:
            raise ValueError("n_max")
        _roam_defaults(self)
        self.o_prep, self.o_exec = float(o_prep), float(o_exec)
        self.ttt_exec_s, self.ttt_prep_s = float(ttt_exec), 0.0
        self.n_max, self.release_hys_db = int(n_max), 3.0
        self.eta_H, self.eta_L, self.eta_A = float(eta_H), float(eta_L), float(eta_A)
        self.eta_S, self.eta_D, self.rho = float(eta_S), float(eta_D), float(rho)
        self.use_memory = bool(memory)
        self.report_offset_db = self.o_prep - 6.0 * self.rho
        self.need_area = self.eta_A != 0.0
        self.need_dwell = self.eta_D != 0.0

    def params(self):
        return dict(o_prep=self.o_prep, o_exec=self.o_exec, ttt_exec=self.ttt_exec_s, n_max=self.n_max,
                    eta_H=self.eta_H, eta_L=self.eta_L, eta_A=self.eta_A, eta_S=self.eta_S, eta_D=self.eta_D,
                    rho=self.rho, memory=self.use_memory)

    def reset(self, ctx):
        super().reset(ctx)
        self._roam_reset(ctx)
        self._D_t = None
        self.trivial = (self.eta_H == 0 and self.eta_L == 0 and self.eta_A == 0 and self.eta_S == 0
                        and self.eta_D == 0)

    def _adjust(self, obs):
        if self._D_t == obs.t:
            return self._D
        U = len(obs.serving)
        if self.trivial:
            D = np.zeros((U, self.M))
        else:
            self._decay(obs.t)
            gam = self._motion(obs)
            self._geometry(obs)
            serv = np.asarray(obs.serving).astype(int)
            D = np.zeros((U, self.M))
            if self.eta_H or self.eta_L:
                H, L = self._scores(serv, self.bins(obs.x, obs.y))
                D += self.eta_H * (H - 0.25) - self.eta_L * (L - 0.25)
            if self.eta_S:
                D += self.eta_S * (self.w_s - self.w_bar)[:, None]
            if self.eta_A:
                fc = self.f[self.ar, serv]
                D -= self.eta_A * gam[:, None] * (self.f - fc[:, None])
            if self.eta_D:
                D += self.eta_D * self.s
            D = np.clip(D, -6.0, 6.0)
        self._D, self._D_t = D, obs.t
        return D

    def prep_offsets(self, obs):
        if self.trivial or self.rho == 0.0:
            if not self.trivial:
                self._adjust(obs)
            return self.o_prep
        return self.o_prep + self.rho * self._adjust(obs)

    def exec_offsets(self, obs):
        if self.trivial:
            return self.o_exec
        oe = np.round((self.o_exec + self._adjust(obs)) * 2.0) / 2.0
        return np.clip(oe, -15.0, 15.0)


class RoamA3(_RoamParts, BaselineRule):
    """ROAM on Event A3 as published: CIO m = h0 + eta_H H - eta_L L + eta_S w_s - eta_A gamma (f_j - f_c),
    clipped to [0, 9] dB, TTT, hold T_hold; epsilon = 0 (no base layer)."""
    learns = True

    def __init__(self, h0=2.0, ttt=0.3, eta_H=14.0, eta_L=8.0, eta_A=8.0, eta_S=4.0, t_hold=1.0,
                 m_min=0.0, m_max=9.0, name="RA3"):
        super().__init__(name)
        _roam_defaults(self)
        self.h0, self.ttt = float(h0), float(ttt)
        self.eta_H, self.eta_L, self.eta_A, self.eta_S = float(eta_H), float(eta_L), float(eta_A), float(eta_S)
        self.t_hold, self.m_min, self.m_max = float(t_hold), float(m_min), float(m_max)
        self.use_memory = True
        self.need_area = self.eta_A != 0.0
        self.need_dwell = False

    def params(self):
        return dict(h0=self.h0, ttt=self.ttt, eta_H=self.eta_H, eta_L=self.eta_L, eta_A=self.eta_A,
                    eta_S=self.eta_S, t_hold=self.t_hold)

    def reset(self, ctx):
        super().reset(ctx)
        self._roam_reset(ctx)
        self.prev_serv = None
        self.t_last = np.full(ctx.n_users, -np.inf)
        self.t_H = -np.inf

    def a3_offsets(self, obs):
        t = obs.t
        serv = np.asarray(obs.serving).astype(int)
        if self.prev_serv is not None:
            ch = serv != self.prev_serv
            self.t_last[ch] = t
        self.prev_serv = serv.copy()
        self._decay(t)
        gam = self._motion(obs)
        U = len(serv)
        m = np.full((U, self.M), self.h0)
        if self.eta_H or self.eta_L:
            if t - self.t_H >= 0.3 - 1e-9 or not hasattr(self, "_HL"):     # H and L refreshed every 300 ms
                self._HL = self._scores(serv, self.bins(obs.x, obs.y))
                self.t_H = t
            H, L = self._HL
            m += self.eta_H * H - self.eta_L * L
        if self.eta_S:
            m += self.eta_S * self.w_s[:, None]
        if self.eta_A:
            self._geometry(obs)
            fc = self.f[self.ar, serv]
            m -= self.eta_A * gam[:, None] * (self.f - fc[:, None])
        m = np.clip(m, self.m_min, self.m_max)
        if self.t_hold > 0:
            m[t - self.t_last < self.t_hold] = np.inf
        return m

    def time_to_trigger(self, obs):
        return self.ttt
