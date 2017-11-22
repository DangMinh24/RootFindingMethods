import numpy as np
import tensorflow as tf

class Newton():
    def __init__(self,sess,func,tensor_x,left_boundary=10e-2,right_boundary=10e2,step=0.2,floating_point=4,iterators=30):
        self.left_boundary=left_boundary
        self.right_boundary=right_boundary
        self.step=step
        self.func=func
        self.roots=set()
        self.valid_intervals=[]
        self.sess=sess
        self.floating_point=floating_point
        self.x=tensor_x
        self.grad=tf.gradients(self.func,self.x)[0]
        self.newtons_iterator=iterators
    def find_valid_intervals(self):

        print("der: %d"%(self.sess.run(self.grad,feed_dict={self.x:1})))
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

        if len(self.valid_intervals) == 0:
            print("This function is always positive or negative")
            print("There is no root to find")
        else:
            print("From global interval (%f,%f), we find out that possible sub-intervals that root can be: " %(self.left_boundary,self.right_boundary),end=" ")
            print(self.valid_intervals)

    def find_roots(self):
        if len(self.valid_intervals) == 0:
            pass
        else:
            for jter,(left,right) in enumerate(self.valid_intervals):
                print("Sub_interval %d"%jter)
                init_x=float(np.random.randint(left*100,right*100)/100) #########Init random value from range(left,right)

                previous_x=init_x
                previous_y=self.sess.run(self.func,feed_dict={self.x:previous_x})
                for iter in range(self.newtons_iterator):
                    print("\tStep %d.%d"%(jter,iter))
                    derivative=self.sess.run(self.grad,feed_dict={self.x:previous_x})

                    if derivative==0:
                        print("There is a case that make derivative = 0. Can't apply Newton's Method for this function ")
                        break
                    next_x=previous_x-previous_y/derivative
                    next_y=self.sess.run(self.func,feed_dict={self.x:next_x})
                    print("\tGuessing root as :%f"%next_x)
                    if round(next_y,self.floating_point)==0:
                        self.roots.add(next_x)
                        break
                    previous_x=next_x
                    previous_y=next_y

