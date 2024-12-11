from networks import PolicyOverOptions, SubPolicy, TerminationPolicy, QOmega, QU
import numpy as np

class OptionsCriticModel():
    def __init__(self, ):
        self.num_options = num_options
        self.num_actions = num_actions

        self.policy_over_options = PolicyOverOptions(4, self.num_options)
        self.sub_policy = SubPolicy(60, self.num_actions)
        self.termination_policy = TerminationPolicy(10)
        self.Qomega = QOmega()
        self.Qu = QU()

        self.current_option = None

        return self.policy_over_options, self.sub_policy, self.termination_policy, self.Qomega, self.Qu

    def get_one_hot_encoding(self, o, n):
        one_hot = np.zeros(n)
        one_hot[o] = 1
        return list(one_hot)

    def forward(self, input):
        if self.current_option == None:
            option_probs = self.policy_over_options(input[-4:])
            option = np.argmax(option_probs)
            self.current_option = option
        
        one_hot_encoded_option = self.get_one_hot_encoding(self.current_option, self.num_options)

        output = self.sub_policy(input[:54]+one_hot_encoded_option)

        is_terminate = self.termination_policy(input[-4:]+one_hot_encoded_option)
        if is_terminate > 0.5:
            self.current_option = None
        return output, is_terminate
        


        