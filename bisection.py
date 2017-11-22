import numpy as np
import tensorflow as tf

class Bisection():
    def __init__(self,sess,func,tensor_x,left_boundary=10e-2,right_boundary=10e2,step=0.2,floating_point=4):
        self.left_boundary=left_boundary
        self.right_boundary=right_boundary
        self.step=step
        self.func=func
        self.roots=set()
        self.valid_intervals=[]
        self.sess=sess
        self.floating_point=floating_point
        self.x=tensor_x
    def find_valid_intervals(self):

        f_j=self.sess.run(self.func,feed_dict={self.x:self.left_boundary})
        j=self.left_boundary
        for i in np.arange(self.left_boundary,self.right_boundary,self.step):
            # print(i)
            f_i=self.sess.run(self.func,feed_dict={self.x:i})
            if f_i==0:
                pass
            elif f_i*f_j<0 and i!=j:
                self.valid_intervals.append((j,i))
                j=i
                f_j=f_i
        if len(self.valid_intervals)==0:
            print("This function is always positive or negative")
            print("There is no root to find")
        else:

            print("From global interval (%f,%f), we find out that possible sub-intervals that root can be: " %(self.left_boundary,self.right_boundary),end=" ")
            print(self.valid_intervals)
    def find_roots(self):
        def recursive(subleft_bound,subright_bound,floating_point,session):
            if round(subleft_bound, floating_point) >= round(subright_bound, floating_point):
                self.roots.add(round(subleft_bound,floating_point))
                return round(subleft_bound, floating_point)
            else :
                average_point=(subleft_bound+subright_bound)/2
                f_a=session.run(self.func,feed_dict={self.x:average_point})
                f_l=session.run(self.func,feed_dict={self.x:subleft_bound})
                f_r=session.run(self.func,feed_dict={self.x:subright_bound})
                if f_a * f_l < 0:
                    recursive(subleft_bound,average_point,floating_point,session)
                elif f_a*f_r <0:
                    recursive(average_point,subright_bound,floating_point,session)
        if len(self.valid_intervals)==0:
            pass
        else:
            for left,right in self.valid_intervals:

                recursive(left,right,self.floating_point,self.sess)
