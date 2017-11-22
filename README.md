# RootFindingMethods

## Abstract:
[Finding root](https://en.wikipedia.org/wiki/Root-finding_algorithm) is a mathematic problem that is vital for deeper applications in various field. However, root of mostly natural functions can not be found exactly
Therefore, there are some techniques for finding value that approximately close to the real root.

## Prerequisite:

  * Install python

    sudo apt-get install python3

  * Install requirement libraries
  
    sudo pip3 install -r requirements

## Preprocessing:

  + Convert string expression into mathematic expression using [postfix](http://interactivepython.org/runestone/static/pythonds/BasicDS/InfixPrefixandPostfixExpressions.html) technique

  + Wrapping mathematic expression with [tensorflow](https://www.tensorflow.org/versions/r0.12/api_docs/python/train/gradient_computation) operators (So that can easily compute derivative of the given functions)

  + Split global interval into possible valid sub-intervals that contain roots. Which can be used as information to prove whether the functions have root (Using [Intermediate Value Theorem](https://en.wikipedia.org/wiki/Intermediate_value_theorem))

## Usage:

      python3 main.py -m 2 -lb -100 -rb 100 -s 0.2 -fp 4 -i 30
  
  * m: for methods. There are 2 methods, 1/ for Bisection, and 2/ for Newton's
  
  * lb, rb: restriction of intervals. Should not be too large (Memory Problem)
  
  * s: step size to search for sub-intervals. Should be change to fit appropriate with users's system
  
  * fp: floating point. Can't get exactly value in some cases (That is the reason why using these techinques). Use floating point to know when the results are acceptable
  
  * i: iterators. This one for Newton's only (-m 2). Number of step that program will update value (xn)
  
  

## Summary:

Use two basic techniques for finding root.

  1. [Bisection](https://en.wikipedia.org/wiki/Bisection_method):

      Use binary search technique for finding approximate root in each sub-intervals (Recursive algorithm)

  2. [Newton's method](https://en.wikipedia.org/wiki/Newton%27s_method)

      Use tangent line at random point(x0) in each sub-intervals and recursively update the point (xn) until figure out the root  
