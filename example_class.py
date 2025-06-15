class A_class_example:
    def __init__(self, text_a, text_b, number_a, number_b):
        self.text_a = text_a
        self.text_b = text_b
        self.number_a = number_a
        self.number_b = number_b

    def combine_string(self):
        combined_strings = text_a + text_b
        return combined_strings

    def combine_numbers(self):
        combined_number = self.number_a + self.number_b
        return combined_number

def non_class_adding(number_1, number_2):
    combo = number_1 + number_2
    return combo
