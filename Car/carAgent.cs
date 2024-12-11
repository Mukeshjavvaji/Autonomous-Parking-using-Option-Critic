using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.SideChannels;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using System.Collections.Generic;
using System.Collections;
using System;

namespace AutonomousParking{
    public class carAgent : Agent{

        private Spawner spawner;
        private CarRayPerception perceptionSensor;
        private carControl control;

        private bool isCollided;

        private int prev_option = -1;

        // private float previous_heuristic = 0f;
        private float posX;
        private float posZ;

        private OptionSideChannel optionChannel;

        void Start(){
            spawner = FindObjectOfType<Spawner>();
            perceptionSensor = FindObjectOfType<CarRayPerception>();
            control = FindObjectOfType<carControl>();
            Debug.Log("Academy initialized: " + Academy.IsInitialized);
            optionChannel = new OptionSideChannel();
            SideChannelManager.RegisterSideChannel(optionChannel);

        }
        public override void OnEpisodeBegin(){
            System.Random rand = new System.Random();
            spawner.ResetVehicles();
            SetReward(0);
            float randomX = (float)(rand.NextDouble() * (4.32f + 6.8f) - 6.8f);
            float randomZ = (float)(rand.NextDouble() * (-20.9 + 29.31) - 29.31);
            float randomAngle = (float)(rand.NextDouble() * (180f + 180f) - 180f);
            posX = randomX;
            posZ = randomZ;
            Vector3 spawnPosition = new Vector3(randomX, 0.0f, randomZ);
            Quaternion spawnRotation = Quaternion.Euler(0, randomAngle, 0);
            transform.position = spawnPosition;
            transform.rotation = spawnRotation;

            perceptionSensor.resetVariables();
            // previous_heuristic = 0f;
            isCollided = false;
        }

        private double euclidianDistance(List<float> pos, List<float> goal){
            double dx = pos[0] - goal[0];
            double dy = pos[1] - goal[1];

            return Math.Sqrt(dx * dx + dy * dy);
        }

        private float CalculateAngularDifference(float angle1, float angle2)
        {
            float difference = (angle2 - angle1 + 180) % 360 - 180;
            return Math.Abs(difference);
        }

        private double get_heuristic(){
            List<float> goal = perceptionSensor.getCurrentGoal();
            List<float> pos = new List<float>();
            pos.Add(transform.position[0]);
            pos.Add(transform.position[2]);
            pos.Add(transform.eulerAngles[1]);
            if(goal.Count == 0){
                return 0.0f;
            } else if(goal.Count == 2){
                return euclidianDistance(pos, goal);
            } else{
                return euclidianDistance(pos, goal)+CalculateAngularDifference(pos[2], goal[2]);
            }
        }

        public override void CollectObservations(VectorSensor sensor){
            List<float> rayVals = perceptionSensor.GetRayDistances();
            sensor.AddObservation(rayVals);
            posX = transform.localPosition[0];
            sensor.AddObservation(posX);
            posZ = transform.localPosition[2];
            sensor.AddObservation(posZ);
            sensor.AddObservation(transform.localRotation.eulerAngles[1]);
            sensor.AddObservation(control.get_torque());
            sensor.AddObservation(control.get_steerAngle());
            sensor.AddObservation(perceptionSensor.isSpotFound());
            sensor.AddObservation(perceptionSensor.isEntranceFound());
            sensor.AddObservation(perceptionSensor.isInsidePLot());
        }

        private void OnCollisionEnter(Collision collision){
            isCollided = true;
        }

        private void OnCollisionExit(Collision collision){
            isCollided = false;
        }

        public override void OnActionReceived(ActionBuffers actions){
            SetReward(0);
            List<float> actionVals = new List<float>();
            var continous = actions.ContinuousActions;
            actionVals.Add(continous[0]);
            actionVals.Add(continous[1]);
            control.PerformAction(actionVals);
            float current_heuristic = (float)get_heuristic();
            float poo_reward = 0f;
            float iop_reward = 0f;
            float termination_reward = 0f;
            if(optionChannel.get_current_option() == 0){
                if(perceptionSensor.isEntranceFound() == 1){
                    AddReward(-0.0001f);
                    poo_reward += -0.0001f;
                }
                if(control.get_torque() == 0f){
                    AddReward(-0.0001f);
                    iop_reward += -0.0001f;
                } else if(control.get_torque() > 0.0f) {
                    AddReward(0.0001f);
                    iop_reward += 0.0001f;
                }
                if(transform.position[0]>posX-0.01f && transform.position[0]<posX+0.01f){
                    if(transform.position[2]>posZ-0.01f && transform.position[2]<posZ+0.01f){
                        AddReward(-0.0001f);
                        iop_reward += -0.0001f;
                    } else {
                        AddReward(0.0001f);
                        iop_reward += 0.0001f;
                    }
                }

                prev_option = 0;
            } else if(optionChannel.get_current_option() == 1){
                if(perceptionSensor.isEntranceFound() == 0){
                    AddReward(-0.0001f);
                    poo_reward += -0.0001f;
                }
                if(control.get_torque() > 0f){
                    AddReward(0.0001f);
                    iop_reward += 0.0001f;
                }
                if(perceptionSensor.isEntranceFound() == 1){
                    if(perceptionSensor.isInsidePLot() == 0){
                        AddReward(0.03f-current_heuristic/5000.0f);
                        iop_reward += 0.03f-current_heuristic/5000.0f;
                    }
                }
                if(prev_option == 0){
                    if(perceptionSensor.isEntranceFound() == 0){
                        termination_reward += -0.0001f;
                    }
                }
                prev_option = 1;
            } else if(optionChannel.get_current_option() == 2){
                if(perceptionSensor.isInsidePLot() == 0){
                    AddReward(-0.0001f);
                    poo_reward += -0.0001f;
                }
                if(perceptionSensor.isSpotFound() == 1){
                    AddReward(-0.0001f);
                    poo_reward += -0.0001f;
                }
                if(control.get_torque() == 0f){
                    AddReward(-0.0001f);
                    iop_reward += -0.0001f;
                } else {
                    AddReward(0.0001f);
                    iop_reward += 0.0001f;
                }
                if(transform.position[0]>posX-0.0001f && transform.position[0]<posX+0.0001f){
                    if(transform.position[2]>posZ-0.0001f && transform.position[2]<posZ+0.0001f){
                        AddReward(-0.0003f);
                        iop_reward += -0.0003f;
                    } else {
                        AddReward(0.0001f);
                        iop_reward += 0.0001f;
                    }
                }
                if(prev_option == 0){
                    if(perceptionSensor.isEntranceFound() == 0){
                        termination_reward += -0.0001f;
                    }
                }
                if(prev_option == 1){
                    if(perceptionSensor.isInsidePLot() == 0){
                        termination_reward += -0.0001f;
                    }
                }
                prev_option = 2;
            } else if(optionChannel.get_current_option() == 3){
                if(perceptionSensor.isSpotFound() == 0){
                    AddReward(-0.0001f);
                    poo_reward += -0.0001f;
                }
                if(perceptionSensor.isInsidePLot() == 0){
                    AddReward(-0.0001f);
                    poo_reward += -0.0001f;
                }
                if(perceptionSensor.isSpotFound() == 1){
                    if(perceptionSensor.isInsidePLot() == 1){
                        AddReward(0.07f-current_heuristic/5000.0f); 
                        iop_reward += 0.07f-current_heuristic/5000.0f;
                    }
                }
                if(prev_option == 0){
                    if(perceptionSensor.isEntranceFound() == 0){
                        termination_reward += -0.0001f;
                    }
                }
                if(prev_option == 1){
                    if(perceptionSensor.isInsidePLot() == 0){
                        termination_reward += -0.0001f;
                    }
                }
                if(prev_option == 2){
                    if(perceptionSensor.isSpotFound() == 1){
                        termination_reward += -0.0001f;
                    }
                }
                prev_option = 3;
            }
            if(isCollided){
                AddReward(-0.0005f);
                iop_reward += -0.0005f;
            }
            // AddReward(-0.00001f);
            List<float> rewards = new List<float>{poo_reward, iop_reward, termination_reward};
            optionChannel.SendListToPython(rewards);
        }
    }
}
