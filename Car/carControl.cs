using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace AutonomousParking
{
    
    public class carControl : MonoBehaviour
    {
        [SerializeField] private WheelCollider FrontLeftcollider, FrontRightcollider, RearLeftcollider, RearRightcollider;
        private float Torque;
        private float steerAngle;

        private float maxTorque = 30f;
        private float maxSteer = 20f;


        IEnumerator SetTorqueNull(){
            yield return new WaitForSeconds(0.1f);
            FrontLeftcollider.brakeTorque = 0f;
            FrontRightcollider.brakeTorque = 0f;
        }
        void Start()
        {
            
        }

        public void PerformAction(List<float> action){
            SetTorque(action[0]);
            SetSteerAngle(action[1]);
            // StopCar(action[2]);
        }

        // Update is called once per frame
        void Update()
        {
            
        }

        void SetTorque(float torque){
            RearLeftcollider.motorTorque = torque*maxTorque;
            RearRightcollider.motorTorque = torque*maxTorque;
            Torque = torque;
        }

        void SetSteerAngle(float angle){
            FrontLeftcollider.steerAngle = angle*maxSteer;
            FrontRightcollider.steerAngle = angle*maxSteer;
            steerAngle = angle;
        }

        void StopCar(float flag){
            if(flag == 1){
                RearLeftcollider.motorTorque = -3000;
                RearRightcollider.motorTorque = -3000;
                StartCoroutine(SetTorqueNull());
            }
        }

        public float get_torque(){
            return Torque;
        }

        public float get_steerAngle(){
            return steerAngle;
        }
    }
}
