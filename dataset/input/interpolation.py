from typing import cast, Optional, TYPE_CHECKING

import equinox as eqx
import equinox.internal as eqxi
import jax.numpy as jnp
import jax.tree_util as jtu


if TYPE_CHECKING:
    from typing import ClassVar as AbstractVar
else:
    from equinox import AbstractVar

from equinox.internal import ω
from jaxtyping import Array, ArrayLike, PyTree, Real, Shaped

from diffrax._custom_types import DenseInfos, IntScalarLike, RealScalarLike, Y
from diffrax._local_interpolation import AbstractLocalInterpolation
from diffrax._misc import fill_forward, left_broadcast_to
from diffrax._path import AbstractPath
from diffrax._global_interpolation import AbstractGlobalInterpolation


class ZOHInterpolation(AbstractGlobalInterpolation):
    """Interpolates some data `ys` over the interval $[t_0, t_1]$ with knots
    at `ts` with a piecewise constant.

    !!! warning

        If using `LinearInterpolation` as part of a [`diffrax.ControlTerm`][], then the
        vector field will make a jump every time one of the knots `ts` is passed. If
        using an adaptive step size controller such as [`diffrax.PIDController`][],
        then this means the controller should be informed about the jumps, so that it
        can handle them appropriately:

        ```python
        ts = ...
        interp = LinearInterpolation(ts=ts, ...)
        term = ControlTerm(..., control=interp)
        stepsize_controller = PIDController(..., jump_ts=ts)
        ```
    """

    ts: Real[Array, " times"]
    ys: PyTree[Shaped[Array, "times ..."]]

    def _interpret_t_zoh(
        self, t: RealScalarLike, left: bool
    ) -> tuple[IntScalarLike, RealScalarLike]:
        maxlen = self.ts_size - 1
        index = jnp.searchsorted(self.ts, t, side="left" if left else "right")
        index = jnp.clip(index - 1, min=0, max=maxlen)
        # Will never access the final element of `ts`; this is correct behaviour.
        fractional_part = t - self.ts[index]
        return index, fractional_part

    def __check_init__(self):
        def _check(_ys):
            if _ys.shape[0] != self.ts.shape[0]:
                raise ValueError(
                    "Must have ts.shape[0] == ys.shape[0], that is to say the same "
                    "number of entries along the timelike dimension."
                )

        jtu.tree_map(_check, self.ys)

    @property
    def ts_size(self) -> IntScalarLike:
        return self.ts.shape[0]

    @eqx.filter_jit
    def evaluate(
        self,
        t0: RealScalarLike,
        t1: Optional[RealScalarLike] = None,
        left: bool = False,
    ) -> PyTree[Array]:
        r"""Evaluate the linear interpolation.

        **Arguments:**

        - `t0`: Any point in $[t_0, t_1]$ to evaluate the interpolation at.
        - `t1`: If passed, then the increment from `t1` to `t0` is evaluated instead.
        - `left`: Across jump points: whether to treat the path as left-continuous
            or right-continuous. [In practice linear interpolation is always continuous
            except around `NaN`s.]

        !!! faq "FAQ"

            Note that we use $t_0$ and $t_1$ to refer to the overall interval, as
            obtained via `instance.t0` and `instance.t1`. We use `t0` and `t1` to refer
            to some subinterval of $[t_0, t_1]$. This is an API that is used for
            consistency with the rest of the package, and just happens to be a little
            confusing here.

        **Returns:**

        If `t1` is not passed:

        The interpolation of the data. Suppose $t_j < t < t_{j+1}$, where $t$ is `t0`
        and $t_j$ and $t_{j+1}$ are some element of `ts` as passed in `__init__`.
        Then the value returned is
        $y_j + (y_{j+1} - y_j)\frac{t - t_j}{t_{j+1} - t_j}$.

        If `t1` is passed:

        As above, with $t$ taken to be both `t0` and `t1`, and the increment between
        them returned.
        """

        if t1 is not None:
            return self.evaluate(t1, left=left) - self.evaluate(t0, left=left)
        index, fractional_part = self._interpret_t_zoh(t0, left)
        # index = index +  (fractional_part >= 1.0)

        def _index(_ys):
            return _ys[index]

        return jtu.tree_map(_index, self.ys)

    @eqx.filter_jit
    def derivative(self, t: RealScalarLike, left: bool = True) -> PyTree[Array]:
        r"""Evaluate the derivative of the linear interpolation. Essentially equivalent
        to `jax.jvp(self.evaluate, (t,), (jnp.ones_like(t),))`.

        **Arguments:**

        - `t`: Any point in $[t_0, t_1]$ to evaluate the derivative at.
        - `left`: Whether to obtain the left-derivative or right-derivative at that
            point.

        **Returns:**

        The derivative of the interpolation of the data. Suppose $t_j < t < t_{j+1}$,
        where $t_j$ and $t_{j+1}$ are some elements of `ts` passed in `__init__`. Then
        the value returned is $\frac{y_{j+1} - y_j}{t_{j+1} - t_j}$.
        """

        index, _ = self._interpret_t(t, left)

        def _zeros(_ys):
            return jnp.zeros_like(_ys[index])

        return jtu.tree_map(_zeros, self.ys)
