"""单个可逆反应的理想溶液平衡组成求解。

对唯一配平后的反应（最小互质正整数系数）

    a1 R1 + a2 R2 + …  ⇌  b1 P1 + b2 P2 + …

给定各参与物种的**正**初始摩尔数 n0_i、溶液体积 V（L）与浓度平衡常数 Kc，
以反应进度 ξ（mol，按配平式书写方向为正）写物料守恒：

    n_R(ξ) = n0_R − a·ξ        （反应物，正向进度被消耗）
    n_P(ξ) = n0_P + b·ξ        （生成物，正向进度被产出）

非负物料区间由各物种初始量与配平系数确定：

    ξ_min = −min_P(n0_P / b) ≤ 0 ≤ ξ_max = min_R(n0_R / a)

因各初始摩尔数均严格为正，必有 ξ_min < 0 < ξ_max，区间为开区间，
端点处至少一个物种浓度为 0。理想溶液中以各物种浓度的计量次方构造

    Qc(ξ) = Π_P (n_P/V)^b  /  Π_R (n_R/V)^a

Qc 在 (ξ_min, ξ_max) 上关于 ξ 严格单调递增（端点处取 0 / +∞），因此
``Qc = Kc``（Kc > 0）在区间内存在**唯一内点**：

* Qc(0) < Kc：反应正向进行，根在 (0, ξ_max)；
* Qc(0) > Kc：反应逆向进行（允许负进度），根在 (ξ_min, 0)；
* Qc(0) = Kc：初始即平衡，ξ = 0。

求解全程采用 :class:`fractions.Fraction` 二分（符号判定逐位精确，绝无
浮点抵消或误判）。每次迭代在中点处同时检验两个终止准则：

* 区间宽度 < 1e-9 mol（中点进度绝对误差 < 5e-10 mol）；
* 反应商相对误差 |Qc(ξ)/Kc − 1| ≤ 1e-8，以精确有理数判定
  |num − Kc·den| ≤ 1e-8·Kc·den（den > 0，diff 与 Qc − Kc 同号）。

第二个准则取代固定相对宽度：极端 Kc（如 1e±100）下唯一内点到物料端点
的距离可小至 1e-34 ~ 1e-50 量级，任何与 Kc 无关的固定宽度都会提前停下，
使 equilibrium_qc 偏离 Kc 数百倍甚至更多；改为按 Qc 相对误差继续逼近后，
正向、逆向都能在非负物料区间内收敛到满足规范（≤ 1e-6）的内点。内部取
1e-8，为输出渲染的舍入误差预留余量，使按响应数值复算的反应商同样达标。

常规数量级约 30~60 次对分，Kc=1e±100 约 200 次，输入数量级上限
（Kc=1e±10000）约 1.2 万次；最终进度与各物种平衡摩尔数均为精确有理数，
仅在 API 渲染层转十进制。
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable

from .balancer import Species
from .cache import species_key

# 二分区间宽度容差：宽度 < 1e-9 mol 时取中点，绝对误差 < 5e-10 mol
WIDTH_TOL = Fraction(1, 10**9)

# 内部反应商相对容差 |Qc(ξ*)/Kc − 1| ≤ 1e-8：极端 Kc 下唯一内点贴近物料
# 端点（残余量可至 1e-50 量级），与 Kc 无关的固定区间宽度会提前停下，使
# equilibrium_qc 偏离 Kc 数百倍甚至更多；按相对误差继续二分可在非负物料
# 区间内收敛。规范要求 ≤ 1e-6，取 1e-8 为输出渲染的舍入误差预留余量
QC_REL_TOL = Fraction(1, 10**8)

# 对分次数保护上限：常规数量级约 30~60 次，Kc=1e±100 约 200 次，
# Kc=1e±10000（输入数量级上限）约 1.2 万次
MAX_BISECTION_ITERS = 100_000


@dataclass(frozen=True)
class EquilibriumSolution:
    """平衡求解结果（进度与各物种平衡摩尔数均为精确 :class:`Fraction`）。"""

    extent: Fraction
    extent_min: Fraction
    extent_max: Fraction
    direction: str  # "forward" / "reverse" / "at_equilibrium"
    initial_qc: Fraction
    equilibrium_qc: Fraction
    equilibrium_moles: dict[str, Fraction]
    iterations: int


def _qc_parts(
    xi: Fraction,
    r_keys: list[str],
    p_keys: list[str],
    coef: dict[str, int],
    n0: dict[str, Fraction],
    volume: Fraction,
) -> tuple[Fraction, Fraction]:
    """Qc = num / den：num = Π n_P^b，den = Π n_R^a · V^Δn。

    正向进度产出生成物（n_P = n0_P + bξ）、消耗反应物（n_R = n0_R − aξ）。
    """
    num = Fraction(1)
    for k in p_keys:
        num *= (n0[k] + coef[k] * xi) ** coef[k]
    den = Fraction(1)
    for k in r_keys:
        den *= (n0[k] - coef[k] * xi) ** coef[k]
    delta_n = sum(coef[k] for k in p_keys) - sum(coef[k] for k in r_keys)
    den *= volume ** delta_n
    return num, den


def _qc_close_enough(diff: Fraction, kc: Fraction, den: Fraction) -> bool:
    """反应商终止准则：|Qc(ξ)/Kc − 1| ≤ 1e-8 的精确有理数判定。

    ``diff = num − Kc·den`` 与 Qc − Kc 同号（区间内 den > 0），故
    |Qc/Kc − 1| = |diff| / (Kc·den)，无需除法即可精确判定。
    """
    return abs(diff) <= QC_REL_TOL * kc * den


def solve_equilibrium(
    reactants: Iterable[Species],
    products: Iterable[Species],
    coefficients: dict[str, int],
    initial: dict[str, Fraction],
    volume: Fraction,
    kc: Fraction,
) -> EquilibriumSolution:
    """求 Qc(ξ) = Kc 的唯一内点进度与各物种平衡摩尔数。

    ``coefficients`` / ``initial`` 均以 :func:`app.cache.species_key` 为键；
    ``initial`` 必须覆盖反应两侧全部物种且值严格为正，``volume`` / ``kc``
    严格为正（合法性由业务层保证）。
    """
    reactants = sorted(reactants, key=lambda sp: sp.formula.key)
    products = sorted(products, key=lambda sp: sp.formula.key)
    r_keys = [species_key(sp) for sp in reactants]
    p_keys = [species_key(sp) for sp in products]
    coef = coefficients
    n0 = initial

    # 非负物料区间：ξ_min = -min_生成物(n0/b)，ξ_max = min_反应物(n0/a)
    extent_min = -min(n0[k] / coef[k] for k in p_keys)
    extent_max = min(n0[k] / coef[k] for k in r_keys)

    num0, den0 = _qc_parts(Fraction(0), r_keys, p_keys, coef, n0, volume)
    initial_qc = num0 / den0

    if initial_qc == kc:
        return _build_solution(
            Fraction(0), extent_min=extent_min, extent_max=extent_max,
            direction="at_equilibrium", initial_qc=initial_qc,
            r_keys=r_keys, p_keys=p_keys, coef=coef, n0=n0, volume=volume,
            iterations=0,
        )

    if initial_qc > kc:
        lo, hi, direction = extent_min, Fraction(0), "reverse"
    else:
        lo, hi, direction = Fraction(0), extent_max, "forward"

    # ---- 精确有理数二分：f = num - Kc·den，与 Qc-Kc 同号且严格递增
    iterations = 0
    for iterations in range(1, MAX_BISECTION_ITERS + 1):
        mid = (lo + hi) / 2
        num, den = _qc_parts(mid, r_keys, p_keys, coef, n0, volume)
        diff = num - kc * den
        if diff == 0:
            extent = mid
            break
        # 终止准则同时满足才收敛：区间宽度 < 1e-9 mol（中点进度绝对误差
        # < 5e-10 mol），且中点反应商相对 Kc 的误差 ≤ 1e-8（极端 Kc 下
        # 继续在非负物料区间内逼近，不按固定宽度提前停下）
        if hi - lo < WIDTH_TOL and _qc_close_enough(diff, kc, den):
            extent = mid
            break
        if diff > 0:
            hi = mid
        else:
            lo = mid
    else:  # pragma: no cover - 保护上限远超任何合法输入所需次数
        extent = (lo + hi) / 2

    return _build_solution(
        extent, extent_min=extent_min, extent_max=extent_max,
        direction=direction, initial_qc=initial_qc,
        r_keys=r_keys, p_keys=p_keys, coef=coef, n0=n0, volume=volume,
        iterations=iterations,
    )


def _build_solution(
    extent: Fraction,
    *,
    extent_min: Fraction,
    extent_max: Fraction,
    direction: str,
    initial_qc: Fraction,
    r_keys: list[str],
    p_keys: list[str],
    coef: dict[str, int],
    n0: dict[str, Fraction],
    volume: Fraction,
    iterations: int,
) -> EquilibriumSolution:
    """按精确有理进度计算各物种平衡摩尔数与平衡 Qc 并组装结果。"""
    moles: dict[str, Fraction] = {}
    for k in r_keys:
        moles[k] = n0[k] - coef[k] * extent
    for k in p_keys:
        moles[k] = n0[k] + coef[k] * extent
    num = Fraction(1)
    for k in p_keys:
        num *= moles[k] ** coef[k]
    den = Fraction(1)
    for k in r_keys:
        den *= moles[k] ** coef[k]
    delta_n = sum(coef[k] for k in p_keys) - sum(coef[k] for k in r_keys)
    den *= volume ** delta_n
    return EquilibriumSolution(
        extent=extent,
        extent_min=extent_min,
        extent_max=extent_max,
        direction=direction,
        initial_qc=initial_qc,
        equilibrium_qc=num / den,
        equilibrium_moles=moles,
        iterations=iterations,
    )
