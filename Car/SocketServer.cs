using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using Newtonsoft.Json;

namespace AutonomousParking
{

    public class SocketServer : MonoBehaviour
    {
        // private SimpleServer server;
        TcpListener tcpListener;
        TcpClient tcpClient;
        NetworkStream networkStream;

        private carControl controller;

        void Start()
        {
            // Initialize the TCPListener to listen for incoming client connections
            tcpListener = new TcpListener(IPAddress.Parse("127.0.0.1"), 65432); // Localhost and port
            tcpListener.Start();
            Debug.Log("Server started and waiting for connection...");

            // Accept the first client that connects
            tcpClient = tcpListener.AcceptTcpClient();
            networkStream = tcpClient.GetStream();  // Get the network stream to communicate with the client
            Debug.Log("Client connected!");

            controller = GetComponent<carControl>();
        }

        public List<int> SendData(List<float> vals)
        {
            // Debug.Log(vals[0]);
            if (networkStream == null)
            {
                Debug.LogError("Network stream is not initialized.");
            }
            string jsonData = JsonConvert.SerializeObject(vals);

            byte[] data = Encoding.ASCII.GetBytes(jsonData);
            List<int> action = new List<int>();
            try
            {

                networkStream.Write(data, 0, data.Length);
                networkStream.Flush();
                // Debug.Log("Data sent");
                action = GetAction();
                // controller.PerformAction(action);
            }
            catch(Exception e)
            {
                Debug.LogError("Error sending data" + e.Message);
            }
            return action;
        }

        public List<int> GetAction()
        {
            byte[] buffer  = new byte[1024];
            int bytesRead = networkStream.Read(buffer, 0, buffer.Length);
            string dataReceived = "None";
            dataReceived = Encoding.UTF8.GetString(buffer, 0, bytesRead);
            List<int> action = JsonConvert.DeserializeObject<List<int>>(dataReceived);
            return action;
        }

        void Update()
        {
            
        } 
    }

}
