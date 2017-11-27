from __future__ import print_function

import hashlib


class TestFailure(Exception):
    pass


class PrivateTestFailure(Exception):
    pass


class Test(object):
    passed = 0
    num_tests = 0
    fail_fast = False
    private = False

    @classmethod
    def setFailFast(cls):
        cls.fail_fast = True

    @classmethod
    def setPrivateMode(cls):
        cls.private = True

    @classmethod
    def assertTrue(cls, result, msg=''):
        cls.num_tests += 1
        if result:
            cls.passed += 1
            print('1 test passed.')
        else:
            print('1 test failed. ' + msg)
            if cls.fail_fast:
                if cls.private:
                    raise PrivateTestFailure(msg)
                else:
                    raise TestFailure(msg)

    @classmethod
    def assertEquals(cls, var, val, msg=''):
        cls.assertTrue(var == val, msg)
        
    @classmethod
    def assertEqualsTol(cls, var, val, tolerance, msg=''):
        equals = abs(var - val) < tolerance
        cls.assertTrue(equals, msg)

    @classmethod
    def assertEqualsHashed(cls, var, hashed_val, msg=''):
        cls.assertEquals(cls._hash(var), hashed_val, msg)

    @classmethod
    def printStats(cls):
        print('{0} / {1} test(s) passed.'.format(cls.passed, cls.num_tests))

    @classmethod
    def _hash(cls, x):
        return hashlib.sha1(str(x)).hexdigest()
