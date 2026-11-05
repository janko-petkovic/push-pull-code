from abc import ABC, abstractmethod
from numpy import ndarray, ones, exp

class ScipyModel(ABC):

    def residual(self, p,x,y):
        return y - self.forward(x,p)

    @abstractmethod
    def forward(self, x,p, *args):
        ...

    @abstractmethod
    def gen_p0(self) -> ndarray :
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

class PowerLaw(ScipyModel):
    def forward(self, x,p,*args):
        return x**p[0]

    def gen_p0(self):
        return ones(1).reshape(1)

    def __str__(self):
        return 'P'

class PowerLawScale(ScipyModel):
    def forward(self, x,p, *args):
        return p[0] * x**p[1]

    def gen_p0(self):
        return ones(2)

    def __str__(self):
        return 'PS'


class PushPullMedian(ScipyModel):
    def forward(self, x,p, *args):
        p = 10**p

        deltak = p[0]
        deltan = p[1]
        sbar = -p[2]
        e50 = p[3]

        return (
            (1 + deltak * e50 * x**(sbar - 1))
            / (1 + deltan * e50 * x**(sbar))
        )

    def gen_p0(self):
        return ones(4)

    def __str__(self):
        return 'PPM'

class PushPullIQ(ScipyModel):

    def forward(self, x,p,*args):
        p = 10**p
        # p = pp
        deltak = p[0]
        deltan = p[1]
        sbar = -p[2]
        e50 = p[3] *exp(args[0])

        return (
            (1 + deltak * e50 * x**(sbar - 1))
            / (1 + deltan * e50 * x**(sbar))
        )

    def gen_p0(self):
        return ones(4)

    def __str__(self):
        return 'PPQ'
