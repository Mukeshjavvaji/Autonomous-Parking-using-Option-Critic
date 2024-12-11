using UnityEngine;
using System.Net;
using System.Net.Sockets;
using System;
using System.Text;
using System.Collections;
using System.Collections.Generic;
using Newtonsoft.Json;

namespace AutonomousParking{

    
    public class SimpleServer{

        TcpListener tcpListener;
        TcpClient tcpClient;
        NetworkStream networkStream;
        
        public void StartServer(){
            
        }

        

        

        void OnApplicationQuit()
        {
            // Clean up by closing the connections when the application quits
            if (networkStream != null)
                networkStream.Close();

            if (tcpClient != null)
                tcpClient.Close();

            if (tcpListener != null)
                tcpListener.Stop();
        }



    }
}