using Unity.MLAgents.SideChannels;
using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;

namespace AutonomousParking
{
    public class OptionSideChannel : SideChannel
    {
        public Action<int> OnOptionReceived;

        private int current_option;

        public OptionSideChannel()
        {
            ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1f7");
        }

        protected override void OnMessageReceived(IncomingMessage msg)
        {
            int optionId = msg.ReadInt32();
            current_option = optionId;
        }

        public void SendListToPython(List<float> values)
        {
            OutgoingMessage msg = new OutgoingMessage();
            msg.WriteInt32(values.Count); // Write the list length
            foreach (float value in values)
            {
                msg.WriteFloat32(value); // Write each float value
            }
            QueueMessageToSend(msg);
        }


        public int get_current_option(){
            return current_option;
        }
    }
}