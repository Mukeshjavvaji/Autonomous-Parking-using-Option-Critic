using UnityEngine;
using Unity.MLAgents.Sensors;
using System.Collections.Generic;
using System.Collections;

namespace AutonomousParking
{
    public class CarRayPerception : MonoBehaviour
    {
        private RayPerceptionSensorComponent3D rayPerceptionSensor;
        public int raysPerDirection = 25;
        public float maxRayDegrees = 180f;
        private List<Vector3> rayDirections;

        private string spot = "ParkingSpot";
        private string entrance = "Entrance";
        private string inside = "Inside";
        private string outside = "Outside";

        private float SpotX = 0.0f;

        private float SpotY = 0.0f;

        private float SpotR = 0.0f;

        private float entranceX = 0.0f;
        private float entranceY = 0.0f;

        private int isInside = 0;

        public void resetVariables(){
            SpotX = 0.0f;
            SpotY = 0.0f;
            SpotR = 0.0f;
            entranceX = 0.0f;
            entranceY = 0.0f;
            isInside = 0;
        }




        void Start()
        {
            
            rayPerceptionSensor = GetComponent<RayPerceptionSensorComponent3D>();
            if (rayPerceptionSensor == null)
            {
                Debug.LogError("Ray Perception Sensor Component is missing on the car.");
            }
            else
            {
                Debug.Log("Sensor Found on GameObject: " + gameObject.name);
                rayDirections = CalculateRayDirections();
            }
            
        }

        void Update(){

        }

        public List<float> GetRayDistances()
        {

            List<float> vals = new List<float>();
            if (rayPerceptionSensor != null && rayDirections != null)
            {
                float rayLength = rayPerceptionSensor.RayLength;
                List<string> detectableTags = rayPerceptionSensor.DetectableTags;
                foreach (var rayDirection in rayDirections)
                {
                    Vector3 worldDirection = transform.TransformDirection(rayDirection);

                    RaycastHit hit;
                    if (Physics.Raycast(transform.position, worldDirection, out hit, rayLength))
                    {
                        if (detectableTags.Contains(hit.collider.tag))
                        {
                            if(hit.collider.tag == spot){
                                GameObject spotObject = hit.collider.gameObject;
                                // Debug.Log("Detected object: " + spotObject.name + " at distance: " + hit.distance);
                                if(SpotX == 0.0){
                                    Debug.Log("Found an empty parking spot "+spotObject);
                                    SpotX = spotObject.transform.localPosition[0];
                                    SpotY = spotObject.transform.localPosition[2];
                                    if(spotObject.name[0] == 'R'){
                                        SpotR = 90;
                                    }else{
                                        SpotR = -90;
                                    }
                                }
                            }
                            if(hit.collider.tag == entrance){
                                GameObject entranceObject = hit.collider.gameObject;
                                // Debug.Log("Detected object: " + entranceObject.name + " at distance: " + hit.distance);
                                if(entranceX == 0.0f){
                                    Debug.Log("Found the entrance");
                                    entranceX = entranceObject.transform.localPosition[0];
                                    entranceY = entranceObject.transform.localPosition[2];
                                }
                            }
                            if(hit.collider.tag == inside){
                                isInside = 1;
                                // Debug.Log("Detected object: inside at distance: " + hit.distance);
                            }
                            if(hit.collider.tag == outside){
                                isInside = 0;
                                // Debug.Log("Detected object: outside at distance: " + hit.distance);
                            }
                        }
                        vals.Add(hit.distance);
                    }
                    else
                    {
                        vals.Add(rayLength);
                    }
                }
            }
            return vals;
        }

        private List<Vector3> CalculateRayDirections()
        {
            List<Vector3> directions = new List<Vector3>();
            float angleIncrement = (2 * maxRayDegrees) / (2*raysPerDirection);
            int halfRays = (raysPerDirection);

            for (int i = -halfRays; i < halfRays; i++)
            {
                float angle = i * angleIncrement;
                Vector3 direction = Quaternion.Euler(0, angle, 0) * Vector3.forward;
                directions.Add(direction);
            }

            return directions;
        }

        public int isSpotFound(){
            if(SpotX == 0.0f){
                return 0;
            }else{
                return 1;
            }
        }

        public int isEntranceFound(){
            if(entranceX == 0.0f){
                return 0;
            }else{
                return 1;
            }
        }

        public int isInsidePLot(){
            return isInside;
        }

        public List<float> getCurrentGoal(){
            List<float> goal = new List<float>();
            if(isInside == 1){
                if(isSpotFound() == 1){
                    goal.Add(SpotX);
                    goal.Add(SpotY);
                    goal.Add(SpotR);
                }
            }else{
                if(isEntranceFound() == 1){
                    goal.Add(entranceX);
                    goal.Add(entranceY);
                    goal.Add(0);
                }
            }
            return goal;
        }
    }
}