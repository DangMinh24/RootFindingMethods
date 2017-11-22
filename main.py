from argparse import ArgumentParser
import pyparsing as pp
from pyparsing import Literal,CaselessLiteral,nums,alphas,Word,Optional,Forward,ZeroOrMore
import tensorflow as tf
import math
import operator
from bisection import Bisection
from newton import Newton


op_stack=[]

point=Literal(".")

e=CaselessLiteral("E")
f_number=Word(nums)+Optional(point+Optional(Word(nums)))+Optional(e+Word("+-"+nums,nums))
var=Literal("x")
pi=CaselessLiteral("PI")

plus = Literal("+")
minus = Literal("-")
mult = Literal("*")
div = Literal("/")
lpar = Literal("(").setParseAction(lambda x:(x[0],"lpar"))
rpar = Literal(")").setParseAction(lambda x:(x[0],"rpar"))
addop = (plus | minus).addParseAction(lambda x:(x[0],"op"))
multop = (mult | div).addParseAction(lambda x:(x[0],"op"))
expop = Literal("^").addParseAction(lambda x:(x[0],"op"))

ident=Word(alphas).addParseAction(lambda x:(x[0],"func"))
expr=Forward()
c= f_number|e|pi|var| (ident+lpar+expr+rpar)

b = c+ZeroOrMore(expop+c)

a=b+ZeroOrMore(multop+b)

expr<< a+ZeroOrMore(addop+a)

example="sin(x^5)+6*x^3-4^2*100*x-200"
result=expr.parseString(example)


def parse_expression(express):
    return expr.parseString(express)

precedence_table={
    "sin":4,
    "cos":4,
    "tan":4,
    "round":4,
    "abs":4,
    "^":4,
    "*":3,
    "/":3,
    "+":2,
    "-":2,
    ")":1,
    "(":1
}


def postfix_convert(parsed_list):
    postfix_result=[]
    stack_operators=[]
    for i in range(len(parsed_list)):
        pointer=parsed_list[i]
        if type(parsed_list[i]) is tuple:
            if pointer[1]=="lpar":
                stack_operators.append(pointer)
            elif pointer[1]=="rpar":
                while len(stack_operators)!=0 and stack_operators[-1][1]!="lpar":
                    appended_op=stack_operators.pop()
                    postfix_result.append(appended_op[0])
                stack_operators.pop()
            else:
                while len(stack_operators)!=0 and precedence_table[stack_operators[-1][0]]>=precedence_table[pointer[0]]:
                    appended_op=stack_operators.pop()
                    postfix_result.append(appended_op[0])
                stack_operators.append(pointer)
        else:
            postfix_result.append(pointer)
    while stack_operators:
        appended_op=stack_operators.pop()
        postfix_result.append(appended_op[0])
    return postfix_result

if __name__=="__main__":
    parser=ArgumentParser(usage="python3 main.py -m <method> -b (lb,rb) -s <inteval_step> -fp <float_point> [-i <newton_iter>]",
                          description="Example:\n"
                                      "\npython3 main.py -m 1 -lb -100 -rb 100 -s 0.2 -fp 4 ")
    parser.add_argument("-m","--methods",nargs='*',help="1:for bisection "
                                              "2:for Newton's method")
    parser.add_argument("-lb","--lbound",help="left bound of interval that the function exists")
    parser.add_argument("-rb","--rbound",help="right bound of interval that the function exists")
    parser.add_argument("-s","--step",help="length of each step in the working interval")
    parser.add_argument("-fp","--floating_point",help="the precision for the result of roots")

    parser.add_argument("-i","--iterator",help="number of step for Newton's method")

    args=parser.parse_args()


    DEFAULT_SETTINGS={"lbound":-10e2,
                      "rbound":10e2,
                      "step":0.2,
                      "floating_point":4,
                      "iterator":30}
    if args.methods[0]!='1' and args.methods[0]!='2':
        raise KeyError("Don't know which method to use")
    else:
        if args.lbound:
            DEFAULT_SETTINGS["lbound"]=float(args.lbound)
        if args.rbound:
            DEFAULT_SETTINGS["rbound"]=float(args.rbound)
        if args.step:
            DEFAULT_SETTINGS["step"]=float(args.step)
        if args.floating_point:
            DEFAULT_SETTINGS["floating_point"]=int(args.floating_point)
        if args.iterator:
            DEFAULT_SETTINGS["iterator"]=int(args.iterator)

        # if args.methods[0]=="1":
        #     print("Option 1: Applying Bisection technique...")
        # else:
        #     print("Option 2: Applying Newton's technique...")

        binary_operator_list = ["+", "-", "*", "/", "^"]
        unary_operator_list = ["sin", "cos", "tan", "round", "abs"]
        operator_map = {
            "+": tf.add,
            "-": tf.subtract,
            "*": tf.multiply,
            "/": tf.div,
            "^": tf.pow,
            "sin": tf.sin,
            "tan": tf.tan,
            "cos": tf.cos,
            "round": tf.tan,
            "abs": tf.abs
        }

        x_tensorflow=tf.placeholder(shape=None,dtype=tf.float32,name="x")
        def formalize_postfix(postfix_result):
            compute_stack = []
            reversed_postfix_result=list(reversed(postfix_result))
            while reversed_postfix_result:
                pointer=reversed_postfix_result.pop()
                if pointer not in binary_operator_list and pointer not in unary_operator_list:
                    compute_stack.append(pointer)
                    # print(compute_stack)
                elif pointer in binary_operator_list :
                    b=compute_stack.pop()
                    a=compute_stack.pop()
                    if a=="x":
                        a=x_tensorflow
                    elif type(a)==str:
                        a=float(a)
                    if b=="x":
                        b=x_tensorflow
                    elif type(b) == str:
                        b = float(b)
                    exp=operator_map[pointer](a,b)
                    compute_stack.append(exp)
                elif pointer in unary_operator_list:
                    a=compute_stack.pop()
                    if a == "x":
                        a = x_tensorflow
                    elif type(a) == str:
                        a = float(a)
                    exp=operator_map[pointer](a)
                    compute_stack.append(exp)
            return compute_stack[0]

        print("#"*20)
        print("Press q to quit, or any button to start")
        key = input()
        while key != "q":
            print("Enter your function y:")
            print("Example: y= 4*x^2+10*x-100")
            print("y=",end=" ")
            express = input()
            try:
                pyparsing_express = parse_expression(express)
                postfix_express = postfix_convert(pyparsing_express)
                func = formalize_postfix(postfix_express)
            except:
                print("The function is INCORRECT format. Try again")
                print()
                print("#"*20)
                print("Press q to quit, or any button to start")
                key = input()
                continue
            pyparsing_express=parse_expression(express)
            postfix_express=postfix_convert(pyparsing_express)
            func= formalize_postfix(postfix_express)
            print("Formalizing function by tensorflow...")

            sess=tf.Session()
            if args.methods[0]=="1":
                print("Option 1: Applying Bisection technique...")
                model=Bisection(sess,func,x_tensorflow,
                                DEFAULT_SETTINGS["lbound"],
                                DEFAULT_SETTINGS["rbound"],
                                DEFAULT_SETTINGS["step"],
                                DEFAULT_SETTINGS["floating_point"])
                model.find_valid_intervals()
                model.find_roots()
            elif args.methods[0]=="2":
                print("Option 2: Applying Newton's technique...")
                model=Newton(sess,func,x_tensorflow,
                             DEFAULT_SETTINGS["lbound"],
                             DEFAULT_SETTINGS["rbound"],
                             DEFAULT_SETTINGS["step"],
                             DEFAULT_SETTINGS["floating_point"],
                             DEFAULT_SETTINGS["iterator"])
                model.find_valid_intervals()
                model.find_roots()
            if len(model.roots)>0:
                print("Approximately root of the function are:")
                for iter,i in enumerate(model.roots):
                    print("root %d:%f"%(iter,i))
            print()
            print("#"*20)
            print("Press q to quit, or any button to start")
            key = input()



