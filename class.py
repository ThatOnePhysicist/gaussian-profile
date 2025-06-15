import functions as fn

class A_class_example:
    def __init__(self, text_a, text_b, number_a, number_b):
        self.text_a = text_a
        self.text_b = text_b
        self.number_a = number_a
        self.number_b = number_b

    def combine_string(self):
        combined_strings = self.text_a + self.text_b
        return combined_strings

    def combine_numbers(self):
        combined_number = self.number_a + self.number_b
        return combined_number

a, b = fn.Funct_Class("this doesn't matter", "nani 1").f_class_i("john", "doe")
c, d = fn.Funct_Class("this also doesn't matter", "nani 2").f_class_i_return("alice", "bob")

print(a)
print(b)

print(c)
print(d)

fn.No_Init_Class().no_init_function("this doesnt have self", "this is jsut inputs")
