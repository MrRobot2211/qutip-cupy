from qutip_cupy import CuPyDense
import numpy as np
import pytest

@pytest.fixture(scope="function", params=((1, 2), (5, 10), (7, 3), (2, 5)))
def shape(request):
    return request.param



def test_conversion_cycle(shape):

    from qutip.core import data

    qutip_dense = data.Dense(np.random.uniform(size=shape))

    tr1 = data.to(CuPyDense, qutip_dense)
    tr2 = data.to(data.Dense, tr1)

    assert (qutip_dense.to_array() == tr2.to_array()).all()


def test_shape(shape):

    cupy_dense = CuPyDense(np.random.uniform(size=shape))

    assert (cupy_dense.shape == shape)


def test_adjoint(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)

    cpdense_adj = CuPyDense(array).adjoint()
    qtpdense_adj = data.Dense(array).adjoint()

    assert (cpdense_adj.to_array() == qtpdense_adj.to_array()).all()



@pytest.mark.parametrize(["matrix", "trace"], [pytest.param([[0, 1],[1, 0]], 0),
                                              pytest.param([[2.j, 1],[1, 1]], 1+2.j)])
def test_trace(matrix, trace):

    cupy_array = CuPyDense(matrix)

    assert cupy_array.trace() == trace

import copy
from pytest_benchmark.fixture import BenchmarkFixture 
class BenchWrap(BenchmarkFixture):
    def __init__(self):
        pass
from pytest_benchmark.stats import Metadata
class Wrapper(object):
    def __init__(self,wrapped_class):
        self.__dict__['wrapped_class'] = wrapped_class

    def __getattr__(self,attr):
        #orig_attr = self.wrapped_class.__getattribute__(attr)
        orig_attr = getattr(self.wrapped_class,attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if result == self.wrapped_class:
                    return self
                return result
            return hooked
        else:
            return orig_attr
    def __setattr__(self,attr,value):
        setattr(self.wrapped_class, attr,value)

    # def __call__(self,*args, **kwargs):
    #     return self.wrapped_class(*args, **kwargs)
    def __call__(self, function_to_benchmark, *args, **kwargs):
        if self._mode:
            self.has_error = True
            raise FixtureAlreadyUsed(
                "Fixture can only be used once. Previously it was used in %s mode." % self._mode)
        try:
            self._mode = 'benchmark(...)'
            return self._raw(function_to_benchmark, *args, **kwargs)
        except Exception:
            self.has_error = True
            raise
    def _raw2(self, function_to_benchmark, *args, **kwargs):
        if self.enabled:
            runner = self._make_runner(function_to_benchmark, args, kwargs)

            duration, iterations, loops_range = self._calibrate_timer(runner)

            # Choose how many time we must repeat the test
            rounds = int(ceil(self._max_time / duration))
            rounds = max(rounds, self._min_rounds)
            rounds = min(rounds, sys.maxsize)

            self.stats = self._make_stats(iterations)
            self.stats.extra_info.update({'device':'all'})
            self.statscpu = self._make_stats(iterations)
            self.statscpu.extra_info.update({'device':'cpu'})
            self.statsgpu = self._make_stats(iterations)
            self.statscpu.extra_info.update({'device':'gpu'})

            self._logger.debug("  Running %s rounds x %s iterations ..." % (rounds, iterations), yellow=True, bold=True)
            results = cp.benchmark(repeat)
            for _,res in zip(XRANGE(rounds),results):
                self.stats.update(res)
                self.statscpu.update(res)
                self.statsgpu.update(res)
            self._logger.debug("  Ran for %ss." % format_time(time.time() - run_start), yellow=True, bold=True)
        
        else:
            function_result = function_to_benchmark(*args, **kwargs)
        return function_result

    def _make_stats(self, iterations):
        bench_stats = Metadata(self, iterations=iterations, options={
            "disable_gc": self._disable_gc,
            "timer": self._timer,
            "min_rounds": self._min_rounds,
            "max_time": self._max_time,
            "min_time": self._min_time,
            "warmup": self._warmup,
        })
        self._add_stats(bench_stats)
        #this makes it difficult to track
        #self.stats = bench_stats
        return bench_stats

import time
@pytest.mark.benchmark(
    min_rounds=5,
    timer=time.time,
)
def test_true_div(shape, benchmark):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)
    
    def divide_by_2(cp_arr):
        return cp_arr /2.
    cup_arr = CuPyDense(array)
    # def benchmark2(*args,**kwargs):
    #     return benchmark(*args,**kwargs)
    
    #print(benchmark.__dict__)
    #benchmark2 = BenchWrap(benchmark.__dict__) 
    
    # benchmark2 = copy.deepcopy(benchmark)
    # benchmark2.__class__ = BenchWrap
    # BenchWrap.__init__(benchmark2, "three")
    #tried this magic with deepcopy but fails to serialize
    
    benchmark2 = Wrapper(benchmark)
    cpdense_tr = benchmark2(divide_by_2, cup_arr)
    qtpdense_tr = data.Dense(array) /2.

    assert (cpdense_tr.to_array() == qtpdense_tr.to_array()).all()


def test_itrue_div(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)

    cpdense_tr = CuPyDense(array).__itruediv__(2.)
    qtpdense_tr = data.Dense(array).__itruediv__(2.)

    assert (cpdense_tr.to_array() == qtpdense_tr.to_array()).all()


def test_mul(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)

    cpdense_tr = CuPyDense(array).__mul__(2.+1.j)
    qtpdense_tr = data.Dense(array).__mul__(2.+1.j)

    assert (cpdense_tr.to_array() == qtpdense_tr.to_array()).all()


def test_matmul(shape):

    from qutip.core import data

    array = np.random.uniform(size=shape) + 1.j*np.random.uniform(size=shape)

    cpdense_tr = CuPyDense(array).__mul__(2.+1.j)
    qtpdense_tr = data.Dense(array).__mul__(2.+1.j)

    assert (cpdense_tr.to_array() == qtpdense_tr.to_array()).all()



# def __call__(self, function_to_benchmark, *args, **kwargs):
#         if self._mode:
#             self.has_error = True
#             raise FixtureAlreadyUsed(
#                 "Fixture can only be used once. Previously it was used in %s mode." % self._mode)
#         try:
#             self._mode = 'benchmark(...)'
#             return self._raw2(function_to_benchmark, *args, **kwargs)
#         except Exception:
#             self.has_error = True
#             raise


#     def _raw2(self, function_to_benchmark, *args, **kwargs):
#         if self.enabled:
#             runner = self._make_runner(function_to_benchmark, args, kwargs)

#             duration, iterations, loops_range = self._calibrate_timer(runner)

#             # Choose how many time we must repeat the test
#             rounds = int(ceil(self._max_time / duration))
#             rounds = max(rounds, self._min_rounds)
#             rounds = min(rounds, sys.maxsize)

#             self.stats = self._make_stats(iterations)
#             self.stats.extra_info.update('device':'all')
#             self.statscpu = self._make_stats(iterations)
#             self.statscpu.extra_info.update('device':'cpu')
#             self.statsgpu = self._make_stats(iterations)
#             self.statscpu.extra_info.update('device':'gpu')

#             self._logger.debug("  Running %s rounds x %s iterations ..." % (rounds, iterations), yellow=True, bold=True)
#             results = cp.benchmark(repeat)
#             for _,res in zip(XRANGE(rounds),results):
#                 self.stats.update(res)
#                 self.statscpu.update(res)
#                 self.statsgpu.update(res)
#             self._logger.debug("  Ran for %ss." % format_time(time.time() - run_start), yellow=True, bold=True)
#         if self.enabled and self.cprofile:
#             profile = cProfile.Profile()
#             function_result = profile.runcall(function_to_benchmark, *args, **kwargs)
#             self.stats.cprofile_stats = pstats.Stats(profile)
#         else:
#             function_result = function_to_benchmark(*args, **kwargs)
#         return function_result

#     def _make_stats(self, iterations):
#         bench_stats = Metadata(self, iterations=iterations, options={
#             "disable_gc": self._disable_gc,
#             "timer": self._timer,
#             "min_rounds": self._min_rounds,
#             "max_time": self._max_time,
#             "min_time": self._min_time,
#             "warmup": self._warmup,
#         })
#         self._add_stats(bench_stats)
#         #this makes it difficult to track
#         #self.stats = bench_stats
#         return bench_stats


# _add_stats  appends new benchmark (bench_stats) that is appended into the benchmarksession
# each element of this list will then be taken by the as_dict and dumped_into_the_json.
# Noote that bench_stats shares the group and extra information from the fixture where it was created
# 
# the basic strategy is to have a cass that inherits from benchmark fixture and replace this methods and in each test
# is initialized from benchmark.__dict__
# 
#  another option is to just pass a timer that return the two times
# each time you append to a different stats 
#
#
#