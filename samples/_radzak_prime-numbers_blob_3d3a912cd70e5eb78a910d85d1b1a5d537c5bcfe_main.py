# import line_profiler
from collections import defaultdict
from functools import wraps
import numpy as np
from math import log
from itertools import compress
import sys
import matplotlib.pyplot as plt
import time
import itertools
import timeit
from functools import partial


timings = defaultdict(lambda: defaultdict(list))


def time_once(f):
    # @wraps(f)
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print('func:%r args:[%r, %r] took: %2.7f sec' % (f.__name__, args, kw, te-ts))
        timings[f.__name__]['x'].append(kw['limit'])  # x axis
        timings[f.__name__]['y'].append(te - ts)  # y axis
        return result
    return timed


def timer(f):
    def timed(*args, **kwargs):
        exec_time = min(timeit.Timer(partial(f, *args, **kwargs)).repeat(repeat=1, number=1))
        timings[f.__name__]['x'].append(kwargs['n'])  # x axis
        timings[f.__name__]['y'].append(exec_time)  # y axis
    return timed


# @timer
# def eratosthenes_sieve_generator(n):
#     def implementation():
#         """
#         Yields primes from 2 to n inclusively.
#         """
#         sieve = [False, False] + [True] * (n - 1)
#         for num, is_prime in enumerate(sieve):
#             if is_prime:
#                 yield num
#                 for i in range(num ** 2, n + 1, num):
#                     sieve[i] = False
#     return list(implementation())


@timer
def eratosthenes_sieve_basic(n):
    """
    Generate a list of primes from 2 to n inclusively.
    """
    sieve = [False, False] + [True] * (n - 1)
    for num, is_prime in enumerate(sieve):
        if is_prime:
            for i in range(num ** 2, n + 1, num):
                sieve[i] = False
    return list(compress(range(0, n + 1), sieve))


# @timer
# def eratosthenes_sieve_store_multiples_generator(n):
#     def implementation():
#         multiples = set()
#         for i in range(2, n + 1):
#             if i not in multiples:
#                 yield i
#                 for j in range(i * i, n + 1, i):
#                     multiples.add(j)
#     return list(implementation())


@timer
def eratosthenes_sieve_list_indices(n):
    """
    Generate a list of booleans from 2 to n inclusively.
    """
    sieve = [False, False] + [True] * (n - 1)
    for num, is_prime in enumerate(sieve):
        if is_prime:
            sieve[num ** 2:: num] = [False] * len(sieve[num ** 2:: num])
    primes = list(compress(range(0, n + 1), sieve))
    return primes


@timer
def eratosthenes_sieve_list_indices_slice_proxy(n):
    sieve = [False, False] + [True] * (n - 1)
    slice_proxy = range(len(sieve))
    for num, is_prime in enumerate(sieve):
        if is_prime:
            sieve[num ** 2:: num] = [False] * len(slice_proxy[num ** 2:: num])
    primes = list(compress(range(0, n + 1), sieve))
    return primes


# @timer
# def eratosthenes_sieve_list_indices_generator(n):
#     def implementation():
#         """
#         Generate a list of booleans from 2 to n inclusively:
#             True -> the number is a prime
#             False -> the number is not a prime
#         """
#         sieve = [False, False] + [True] * (n - 1)
#         for num, is_prime in enumerate(sieve):
#             if is_prime:
#                 yield num
#                 sieve[num ** 2:: num] = [False] * len(sieve[num ** 2:: num])
#     return list(implementation())


@timer
def eratosthenes_sieve_list_indices_mathematical(n):
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for num, is_prime in enumerate(sieve):
        if is_prime:
            sieve[num ** 2:: num] = [False] * ((n-num ** 2)//num + 1)
    return list(compress(range(0, n + 1), sieve))


@timer
def eratosthenes_only_odd(n):
    size = n // 2
    sieve = [True] * size
    for i in range(1, int(n ** .5) + 1):
        step = 2 * i + 1
        if sieve[i]:
            sieve[i + step::step] = [False] * (((size - 1) - i) // step)
    primes = list(itertools.compress(range(1, n + 1, 2), sieve))
    primes[0] = 2
    return primes


@timer
def vprimes(maximum=10**6):
    maxidx = maximum//2
    sieve = [True] * maxidx  # 2, 3, 5, 7... might be prime
    j = 0
    for i in range(1, int(maxidx**0.5)+1):
        j += 4*i
        if sieve[i]:
            step = 2*i + 1
            sieve[j::step] = [False] * -(-(maxidx-j) // step)
    # compress sieve
    primes = list(compress(range(1, maximum, 2), sieve))
    primes[0] = 2
    return primes


@timer
def vprimes_2_switch(maximum=10**6):
    maxidx = maximum//2
    sieve = [True] * maxidx  # 2, 3, 5, 7... might be prime
    j = 0
    for i in range(1, int(maxidx**0.5)+1):
        j += 4*i
        if sieve[i]:
            step = 2*i + 1
            sieve[j::step] = [False] * -(-(maxidx-j) // step)
    # compress sieve
    primes = list(compress(range(1, maximum, 2), sieve))
    primes[0] = 2
    return primes


@timer
def rwh_primes(n):
    # http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    sieve = [True] * n
    for i in range(3, int(n**0.5)+1, 2):
        if sieve[i]:
            sieve[i*i::2*i] = [False]*((n-i*i-1)//(2*i)+1)
    return [2] + [i for i in range(3, n, 2) if sieve[i]]


@timer
def rwh_primes2(n):
    # http://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    correction = (n % 6 > 1)
    n = {0: n, 1: n - 1, 2: n + 4, 3: n + 3, 4: n + 2, 5: n + 1}[n % 6]
    sieve = [True] * (n // 3)
    sieve[0] = False
    for i in range(int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[((k * k) // 3)::2 * k] = [False] * ((n // 6 - (k * k) // 6 - 1) // k + 1)
            sieve[(k * k + 4 * k - 2 * k * (i & 1)) // 3::2 * k] = [False] * (
            (n // 6 - (k * k + 4 * k - 2 * k * (i & 1)) // 6 - 1) // k + 1)
    return [2, 3] + [3 * i + 1 | 1 for i in range(1, n // 3 - correction) if sieve[i]]


@timer
def get_primes_erat(n):
    """
    http://archive.oreilly.com/pub/a/python/excerpt/pythonckbk_chap1/index1.html?page=last
    """
    def erat():
        D = {}
        yield 2
        for q in itertools.islice(itertools.count(3), 0, None, 2):
            p = D.pop(q, None)
            if p is None:
                D[q * q] = q
                yield q
            else:
                x = p + q
                while x in D or not (x & 1):
                    x += p
                D[x] = p
    return list(itertools.takewhile(lambda p: p < n, erat()))


@timer
def sundaram3(max_n):
    """
    Author: jbochi
    """
    numbers = list(range(3, max_n+1, 2))
    half = (max_n)//2
    initial = 4

    for step in range(3, max_n+1, 2):
        for i in range(initial, half, step):
            numbers[i-1] = 0
        initial += 2*(step+1)

        if initial > half:
            return [2] + list(filter(None, numbers))


@timer
def ambi_sieve(n):
    # http://tommih.blogspot.com/2009/04/fast-prime-number-generator.html
    s = np.arange(3, n, 2)
    for m in range(3, int(n ** 0.5) + 1, 2):
        if s[(m - 3) // 2]:
            s[(m * m - 3) // 2::m] = 0
    return np.r_[2, s[s > 0]]


@timer
def primesfrom3to(n):
    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    """ Returns a array of primes, p < n """
    assert n >= 2
    sieve = np.ones(n // 2, dtype=np.bool)
    for i in range(3, int(n ** 0.5) + 1, 2):
        if sieve[i // 2]:
            sieve[i * i // 2::i] = False
    return np.r_[2, 2 * np.nonzero(sieve)[0][1::] + 1]


@timer
def primesfrom2to(n):
    # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n-in-python/3035188#3035188
    """ Input n>=6, Returns a array of primes, 2 <= p < n """
    sieve = np.ones(n // 3 + (n % 6 == 2), dtype=np.bool)
    sieve[0] = False
    for i in range(int(n ** 0.5) // 3 + 1):
        if sieve[i]:
            k = 3 * i + 1 | 1
            sieve[((k * k) // 3)::2 * k] = False
            sieve[(k * k + 4 * k - 2 * k * (i & 1)) // 3::2 * k] = False
    return np.r_[2, 3, ((3 * np.nonzero(sieve)[0] + 1) | 1)]


def main():
    for limit in [10 ** power for power in range(5, 9)]:
        # moje sita eratosthenesa
        # eratosthenes_sieve_generator(n=limit)
        # eratosthenes_sieve_basic(n=limit)
        # eratosthenes_sieve_list_indices(n=limit)
        # eratosthenes_sieve_list_indices2(n=limit)
        # eratosthenes_sieve_list_indices_generator(n=limit)
        # eratosthenes_sieve_list_indices_mathematical(n=limit)
        # eratosthenes_nowy2(n=limit)
        eratosthenes_only_odd(n=limit)

        # https://gist.github.com/Veedrac/9700563
        # vprimes(n=limit)
        # vprimes_2_switch(n=limit)
        # rwh_primes(n=limit)
        # rwh_primes2(n=limit)

        # https://stackoverflow.com/questions/2068372/fastest-way-to-list-all-primes-below-n/
        # get_primes_erat(n=limit)
        # sundaram3(n=limit)
        primesfrom2to(n=limit)
        primesfrom3to(n=limit)
        ambi_sieve(n=limit)


def plot():
    for name, timing in timings.items():
        plt.plot(timing['x'], timing['y'], label=name)
    plt.xscale('log')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
    plot()
