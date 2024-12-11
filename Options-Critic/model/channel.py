from mlagents_envs.side_channel.side_channel import SideChannel, IncomingMessage, OutgoingMessage
import uuid

class OptionSideChannel(SideChannel):

    def __init__(self):
        super().__init__(uuid.UUID("621f0a70-4f87-11ea-a6bf-784f4387d1f7"))
        self.poo_reward = 0
        self.iop_reward = 0
        self.termination_reward = 0

    def send_active_option(self, option_id):
        # Create an OutgoingMessage to send the active option
        msg = OutgoingMessage()
        msg.write_int32(option_id)
        super().queue_message_to_send(msg)

    def on_message_received(self, msg: IncomingMessage) -> None:
        # Read the list of floats
        num_values = msg.read_int32()
        values = [msg.read_float32() for _ in range(num_values)]
        # print(f"Received values from Unity: {values}")
        self.poo_reward = values[0]
        self.iop_reward = values[1]
        self.termination_reward = values[2]
