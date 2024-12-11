using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace AutonomousParking
{
    public class Spawner : MonoBehaviour
    {
        [SerializeField] private GameObject prefabtoSpawn1, prefabtoSpawn2;
        [SerializeField] private GameObject prefabtoSpawn3, prefabtoSpawn8;
        [SerializeField] private GameObject prefabtoSpawn4, prefabtoSpawn9;
        [SerializeField] private GameObject prefabtoSpawn5;
        [SerializeField] private GameObject prefabtoSpawn6;
        [SerializeField] private GameObject prefabtoSpawn7;

        private GameObject prefabtoSpawn;

        private List<GameObject> SpawndedCars = new List<GameObject>();

        private List<GameObject> prefabs = new List<GameObject>();

        void Start(){

        }

        private void RemoveExistingCars(){
            foreach(GameObject obj in SpawndedCars){
                Destroy(obj);
            }
            SpawndedCars.Clear();

        }

        private void spawnCar(GameObject cartospawn, Vector3 position, Quaternion rotation){
            GameObject obj = Instantiate(cartospawn, position, rotation);
            Rigidbody rb = obj.GetComponent<Rigidbody>();
            rb.position = position;
            rb.velocity = Vector3.zero;
            SpawndedCars.Add(obj);
            AddBoxCollider(cartospawn);
        }

        private IEnumerator DelayedFunction()
        {
            yield return new WaitForSeconds(0.3f);
            SpawnVehicles();
            // Your code here
        }

        public void ResetVehicles(){
            RemoveExistingCars();
            StartCoroutine(DelayedFunction());
            // SpawnVehicles();
        }
        // Start is called before the first frame update
        private void SpawnVehicles()
        {
            prefabs.Add(prefabtoSpawn1);
            prefabs.Add(prefabtoSpawn2);
            prefabs.Add(prefabtoSpawn3);
            prefabs.Add(prefabtoSpawn4);
            prefabs.Add(prefabtoSpawn5);
            prefabs.Add(prefabtoSpawn6);
            prefabs.Add(prefabtoSpawn7);
            prefabs.Add(prefabtoSpawn8);
            prefabs.Add(prefabtoSpawn9);
            
            int[] sides = { 0, 1 };
            int[] spots = { 0, 1, 2, 3, 4, 5 };
            int randomSide = GetRandomChoice<int>(sides);
            int randomSpot = GetRandomChoice<int>(spots);
            Vector3 spawnPosition = new Vector3(-6.98f, 0.2f, 1.58f);
            Quaternion spawnRotation = Quaternion.Euler(0, -90, 0);
            for (int i = 0; i < 2; i++)
            {
                for (int j = 0; j < 6; j++)
                {
                    if (randomSide == i)
                    {
                        if (randomSpot != j)
                        {
                            prefabtoSpawn = GetRandomChoice(prefabs);
                            spawnCar(prefabtoSpawn, spawnPosition, spawnRotation);
                        }
                        
                    }
                    else
                    {
                        prefabtoSpawn = GetRandomChoice(prefabs);
                        spawnCar(prefabtoSpawn, spawnPosition, spawnRotation);
                        
                    }
                    spawnPosition[2] -= 2.47f;
                }
                
                spawnPosition = new Vector3(4.23f, 0.2f, 1.58f);
                spawnRotation = Quaternion.Euler(0, 90, 0);
            }
        }

        private static T GetRandomChoice<T>(T[] items)
        {
            int index = Random.Range(0, items.Length);
            return items[index];
        }

        private static T GetRandomChoice<T>(List<T> items)
        {
            int index = Random.Range(0, items.Count);
            return items[index];
        }

        private void AddBoxCollider(GameObject car){
            BoxCollider boxCollider = car.GetComponent<BoxCollider>();
            boxCollider.center = new Vector3(0, 0.6304449f, 0);
            boxCollider.size = new Vector3(1.576153f, 1.258026f, 3.916647f);
        }

        // Update is called once per frame
        void Update()
        {
        
        }
    }
}
// using System.Collections.Generic;
// using UnityEngine;

// namespace AutonomousParking
// {
//     public class Spawner : MonoBehaviour
//     {
//         [SerializeField] private List<GameObject> prefabsToSpawn;
//         private List<GameObject> spawnedCars = new List<GameObject>();

//         private Vector3 firstRowStart = new Vector3(-6.98f, 0, 1.58f); // Starting position for the first row
//         private Vector3 secondRowStart = new Vector3(4.23f, 0, 1.58f); // Starting position for the second row
//         private float spotSpacing = 2.47f; // Space between parking spots
//         private int spotsPerRow = 6; // Number of spots per row

//         private int emptySpotIndex; // Index of the random empty spot

//         void Start()
//         {
//             Random.InitState(System.DateTime.Now.Millisecond);  // Ensures randomness based on the current time
//             SpawnVehicles();
//         }

//         private void RemoveExistingPrefabs()
//         {
//             foreach (GameObject obj in spawnedCars)
//             {
//                 Destroy(obj);
//             }
//             spawnedCars.Clear();
//         }

//         public void SpawnVehicles()
//         {
//             RemoveExistingPrefabs();

//             // Randomize the empty spot index for this episode
//             emptySpotIndex = Random.Range(0, spotsPerRow * 2); // 12 spots in total (2 rows of 6 spots)

//             // Log the empty spot index for debugging
//             // Debug.Log($"Empty spot index for this episode: {emptySpotIndex}");

//             // Loop through both rows (2 rows, 6 spots each)
//             for (int row = 0; row < 2; row++)
//             {
//                 Vector3 spawnPosition = row == 0 ? firstRowStart : secondRowStart;
//                 Quaternion spawnRotation = row == 0 ? Quaternion.Euler(0, -90, 0) : Quaternion.Euler(0, 90, 0);

//                 for (int spot = 0; spot < spotsPerRow; spot++)
//                 {
//                     int currentSpotIndex = row * spotsPerRow + spot;

//                     // Skip the randomized empty spot
//                     if (currentSpotIndex == emptySpotIndex)
//                     {
//                         spawnPosition.z -= spotSpacing;
//                         continue;
//                     }

//                     // Choose a random prefab
//                     GameObject prefabToSpawn = GetRandomChoice(prefabsToSpawn);

//                     // Instantiate the car prefab at the calculated position and rotation
//                     GameObject spawnedCar = Instantiate(prefabToSpawn, spawnPosition, spawnRotation);
//                     spawnedCars.Add(spawnedCar);

//                     // Add BoxCollider to the spawned car if it doesn't already have one
//                     AddBoxCollider(spawnedCar);

//                     // Move the spawn position for the next spot
//                     spawnPosition.z -= spotSpacing;
//                 }
//             }
//         }

//         private static T GetRandomChoice<T>(List<T> items)
//         {
//             int index = Random.Range(0, items.Count);
//             return items[index];
//         }

//         private void AddBoxCollider(GameObject car)
//         {
//             BoxCollider boxCollider = car.GetComponent<BoxCollider>();
//             if (boxCollider == null)
//             {
//                 boxCollider = car.AddComponent<BoxCollider>();
//             }
//             boxCollider.center = new Vector3(0, 0.6304449f, 0);
//             boxCollider.size = new Vector3(1.576153f, 1.258026f, 3.916647f);
//         }
//     }
// }