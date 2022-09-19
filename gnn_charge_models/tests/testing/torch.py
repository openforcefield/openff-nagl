import functools

from torch.testing import assert_close

assert_equal = functools.partial(assert_close, rtol=0, atol=0)
