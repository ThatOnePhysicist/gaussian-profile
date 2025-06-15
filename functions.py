import example_class

class Funct_Class:
    def __init__(self, class_input_1, class_input_2):
        self.class_input_1 = class_input_1
        self.class_input_2 = class_input_2
    def f_class_i(self, f_input_1, f_input_2):
        print(f"""This is from the Funct_Class class with an init function.
The first input for the f_class_i function is: {f_input_1},
the second input for the f_class_i function is: {f_input_2}

""")
        return f_input_1, f_input_2
    def f_class_i_return(self, f_input_3, f_input_4):
        print("This function returns the inputs in funct_class\n\n")
        return self.class_input_1, self.class_input_2

class No_Init_Class:
    def no_init_function(self, a, b):
        print(f"""This is from the No_Init_Class class with no init function.
The first input for the no_init_function is {a},
the second input for the no_init_function is {b}.

""")

def function_a():
    print("""This is function_a and is not in a class.
        There is no input for this function.

        """)

def function_b(input_1, input_2):
    print(f"""
        This function is function_b and is not in a class.
        The first input is {input_1}, the second input is {input_2}

        """)
