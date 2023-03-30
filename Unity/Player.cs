using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using indoorMobility.Scripts.Utils;

using UnityEngine.AI;

namespace indoorMobility.Scripts.Game
{
    public class Player : MonoBehaviour
    {
        //[SerializeField] private AppData appData;
        private AppData appData;
        private Environment environment;
        private Camera camera;
        public Rigidbody m_Rigidbody;

        public int error_bump=0;
        public Vector3 agent_position1;
        Vector3 agent_position2;
        public GameObject props;
        public GameObject player;
        public Transform[] objectList;
        public Collider player_collider;
        public Vector3 player_size;

        public Collider object_collider;
        public Vector3 object_size;
        public float x_bound;
        public float z_bound;
        Vector3 target_location;
        List<Transform> targets_list = new List<Transform>();
        public Transform target;

        public bool target_as_object;

        public GameObject[] floor_list;
        public GameObject floor;
        public Collider floor_collider;
        public Vector3 floor_center;
        public Vector3 floor_size;

        private GameObject target_object;
        private int _stepCount;
        private string _collidedWith;
        private int _maxSteps;
        public int StepCount { get => _stepCount;}
        public string CollidedWith { get => _collidedWith;}

        public float finaldistancetotarget;
        public float distance_travelled;
        
        string distance_type;
        public NavMeshPath path;
        public NavMeshPath path_final;
        private NavMeshAgent nav_agent;

        float distance_beforestep;
        float distance_afterstep;
        int episode=0;
        float scale = 0.2f;
        public int max_episodes=100;
        public Vector3 agent_location;
        public bool validation;
        public int chosen_floor;
        public float initialdistancetotarget;

    
        
        private void OnCollisionStay(Collision collision) 
        {  //Test for collisions (automatic Unity process, runs at fixed update)
            _collidedWith = collision.gameObject.name;

            ContactPoint contact = collision.GetContact(0);

            error_bump = 1;
            // _StepCount++;
            Vector3 contact_point = contact.point;
            // print("contactpoint "+contact_point+ _collidedWith);

            // LOG
            // Debug.Log("player hit by");
            // Debug.Log(_collidedWith);

            if (_collidedWith.Contains("Wall"))
            {
                environment.Reward = appData.WallBumpReward;
                // print("wallcollision");
                environment.End    = 2;
            }
            else
            {
                environment.Reward = appData.BoxBumpReward;
                // print("boxcollision");
                environment.End    = 1;
            }


            if  (_stepCount >= _maxSteps) 
            {
                // can be commented out if  finaldistancetotarget = distance_afterstep always true
                
                // if (distance_type == "euclidian")
                // {
                //     // finaldistancetotarget = Vector3.Distance(target_location, agent_position2); //hesaplamalari resete tasi
                //     Vector3 distance_xz;
                //     distance_xz = target_location- agent_position2;
                //     distance_xz.y=0;
                //     finaldistancetotarget = distance_xz.magnitude;
                // }
                // if (distance_type == "manhattan")
                // {
                //     finaldistancetotarget = Mathf.Abs(target_location.x-agent_position2.x) + Mathf.Abs(target_location.z-agent_position2.z);
                // }
                // if (distance_type == "shortest")
                // {
                //     path_final = new NavMeshPath();
                //     // path_final.ClearCorners();
                //     bool path_existence_final;
                //     path_existence_final = NavMesh.CalculatePath(m_Rigidbody.position, target_location, NavMesh.AllAreas, path_final);
                //     print("path status "+  path_final.status);

                //     if ( path_final.status != NavMeshPathStatus.PathInvalid)
                //     {
                //         float lng_final = 0.0f;

                //         for ( int i = 0; i < path_final.corners.Length - 1; i++ )
                //         {
                //             lng_final += Vector3.Distance(path_final.corners[i], path_final.corners[i + 1]);
                //             // comment when running!
                //             // Debug.DrawLine(path_final.corners[i], path_final.corners[i + 1], Color.blue, 5f, false);
                //         }
                //         finaldistancetotarget = lng_final;
                //     }
                // }
                
                // can be commented out until here if  finaldistancetotarget = distance_afterstep always true, by commenting our below line

                finaldistancetotarget = distance_afterstep; 
                // print("final distance to target "+finaldistancetotarget + "afterstep" + distance_afterstep);
                finaldistancetotarget=10*finaldistancetotarget;
                finaldistancetotarget=Mathf.Round(finaldistancetotarget);
                // print("final distance to target "+finaldistancetotarget);
                environment.DistancetoTarget = (byte)finaldistancetotarget; //todo
                // print("distancetotarget=" + finaldistancetotarget);
                // print("TARGET MAXSTEPREACHED");
                
                environment.End=3;
            }

            if (target_as_object)
            {
                // if (target_name == _collidedWith)
                // {
                // print("targetcollided");

                // exact agent size overlap
                // if(_collidedWith == "Sphere" || target_location.x-target_x*0.5 <=contact_point.x && contact_point.x <=target_location.x+target_x*0.5 && target_location.z-target_z*0.5<=contact_point.z && contact_point.z <=target_location.z+target_z*0.5)
                // agent size x2 overlap
                if(_collidedWith == "Sphere" || target_location.x-x_bound <=contact_point.x && contact_point.x <=target_location.x+x_bound && target_location.z-z_bound<=contact_point.z && contact_point.z <=target_location.z+z_bound)
                {
                    environment.Reward = appData.TargetReachedReward;
                    byte k;
                    k=appData.TargetReachedReward;

                    // print("TARGETreached" + contact_point + " " + target_location + " reward "+ k);

                    finaldistancetotarget=0;
                    environment.DistancetoTarget = (byte)finaldistancetotarget;
                     // UnityEngine.Object.DestroyImmediate(target_object);
                    
                    
                    environment.End = 4; // process this on python side so it's counted and new target is created
                    // print("resetting target_as_object with collision");   
                }              
            }        
        }


        public void Move(int action)
        {
            environment.End = 0;
            _collidedWith = "";
            error_bump=0;
            agent_position1=m_Rigidbody.position; //necessary for avoiding resetting agent location after collisions

            distance_beforestep = distance_afterstep; 

            switch (action)
            {
                case 0: // forward
                    {
                        m_Rigidbody.MovePosition(m_Rigidbody.position+transform.forward*appData.ForwardSpeed);
                        agent_position2=m_Rigidbody.position;

                        if (distance_type=="euclidian")
                        {
                            Vector3 distance_xz2;
                            distance_xz2 = target_location- agent_position2;
                            distance_xz2.y=0;
                            distance_afterstep = distance_xz2.magnitude;

                        }

                        if (distance_type=="manhattan")
                        {
                            distance_afterstep = Mathf.Abs(target_location.x-agent_position2.x) + Mathf.Abs(target_location.z-agent_position2.z);
                        }

                        if (distance_type=="shortest")
                        {
                            path = new NavMeshPath();
                            // path_final.ClearCorners();
                            bool path_existence;
                            path_existence = NavMesh.CalculatePath(agent_position2, target_location, NavMesh.AllAreas, path);
                            // path_existence_final = NavMesh.CalculatePath(m_Rigidbody.position, target_location, NavMesh.AllAreas, path_final);
                            print("path status move "+  path.status);

                            if (path.status != NavMeshPathStatus.PathInvalid)
                            {
                                float lng = 0.0f;

                                for ( int i = 0; i < path.corners.Length - 1; i++ )
                                {
                                    lng += Vector3.Distance(path.corners[i], path.corners[i + 1]);
                                    // comment when running!
                                    // Debug.DrawLine(path.corners[i], path.corners[i + 1], Color.yellow, 5f, false);
                                }

                                distance_afterstep = lng;
                                finaldistancetotarget = 10*lng;
                                finaldistancetotarget=Mathf.Round(finaldistancetotarget);
                                environment.DistancetoTarget = (byte) finaldistancetotarget;
                        
                            }
                            // else 
                            // {
                            //     Vector3 distance_xz2;
                            //     distance_xz2 = target_location- agent_position2;
                            //     distance_xz2.y=0;
                            //     distance_afterstep = distance_xz2.magnitude;
                            //     finaldistancetotarget = 10*distance_afterstep;
                            //     finaldistancetotarget=Mathf.Round(finaldistancetotarget);
                            //     environment.DistancetoTarget = (byte) finaldistancetotarget;

                            // }
                        }

                        float distance_approached;
                        distance_approached = distance_beforestep - distance_afterstep;
                        // print("distance_approached " + distance_approached);
                        distance_approached = (float) System.Math.Round(distance_approached,2);

                        float processed_reward;

                        if (distance_afterstep<distance_beforestep)
                        {
                            processed_reward =  appData.ForwardStepReward -100;
                            environment.Reward =  (byte) processed_reward;
                        }

                        else 
                        {
                            print("approach increased" + distance_approached);
                            environment.Reward= (byte) appData.ForwardStepReward;
                        }

                        // exact agent size overlap
                        // if (m_Rigidbody.position.x-player_collider.bounds.size.x*0.5f<target_location.x &&target_location.x< m_Rigidbody.position.x+player_collider.bounds.size.x*0.5f && m_Rigidbody.position.z-player_collider.bounds.size.z*0.5f < target_location.z && target_location.z  < m_Rigidbody.position.z+player_collider.bounds.size.z*0.5f)
                        // agent size x2 overlap
                        if (m_Rigidbody.position.x-player_collider.bounds.size.x<target_location.x &&target_location.x< m_Rigidbody.position.x+player_collider.bounds.size.x&& m_Rigidbody.position.z-player_collider.bounds.size.z < target_location.z && target_location.z  < m_Rigidbody.position.z+player_collider.bounds.size.z)
                        {
                            finaldistancetotarget=0;
                            environment.DistancetoTarget =  (byte) finaldistancetotarget;
                            // print("TARGETREACHED");
                            environment.Reward=appData.TargetReachedReward;
                            // UnityEngine.Object.DestroyImmediate(target_object);
                            environment.End = 4;
                        }
                        break;


                    }

                case 1: // agent wants to rotate left by 90 degrees
                    {
                        transform.Rotate(new Vector3(0f, -90.0f, 0f));
                        // transform.Rotate(new Vector3(0f, -45.0f, 0f));
                        environment.Reward = (byte) appData.RotationReward;

                        break;
                    }

                case 2: // agent wants to rotate right by 90 degrees
                    {
                        transform.Rotate(new Vector3(0f, 90.0f, 0f));
                        // transform.Rotate(new Vector3(0f, 45.0f, 0f));
                        environment.Reward = (byte) appData.RotationReward;

                        break;
                    }
                default:
                    break; // No action 
            }

            _stepCount++;

            if (_stepCount >= _maxSteps) 
            {
                // can be commented out if  finaldistancetotarget = distance_afterstep always true
                
                // if (distance_type == "euclidian")
                // {
                //     // finaldistancetotarget = Vector3.Distance(target_location, agent_position2); //hesaplamalari resete tasi
                //     Vector3 distance_xz;
                //     distance_xz = target_location- agent_position2;
                //     distance_xz.y=0;
                //     finaldistancetotarget = distance_xz.magnitude;
                // }

                // if (distance_type == "manhattan")
                // {
                //     finaldistancetotarget = Mathf.Abs(target_location.x-agent_position2.x) + Mathf.Abs(target_location.z-agent_position2.z);
                // }

                // if (distance_type == "shortest")
                // {
                //     path_final = new NavMeshPath();
                //     // print("path"+path);
                //     // path_final.ClearCorners();
                //     bool path_existence_final;
                //     path_existence_final = NavMesh.CalculatePath(m_Rigidbody.position, target_location, NavMesh.AllAreas, path_final);
                //     print("path status "+  path_final.status);

                //     if ( path_final.status != NavMeshPathStatus.PathInvalid)
                //     {
                //         float scale_final = 0.0f;

                //         for ( int i = 0; i < path_final.corners.Length - 1; i++ )
                //         {
                //             lng_final += Vector3.Distance(path_final.corners[i], path_final.corners[i + 1]);
                //             // comment when running!
                //             // Debug.DrawLine(path_final.corners[i], path_final.corners[i + 1], Color.blue, 5f, false);
                //         }
                //         finaldistancetotarget = lng_final;
                //     }
                // }
                
                // can be commented out until here if  finaldistancetotarget = distance_afterstep always true, by commenting our below line

                finaldistancetotarget = distance_afterstep; 
                finaldistancetotarget=10*finaldistancetotarget;
                finaldistancetotarget=Mathf.Round(finaldistancetotarget);
                print("final distance to target "+finaldistancetotarget);
                environment.DistancetoTarget = (byte)finaldistancetotarget;
               
                environment.End = 3;
            }
        }

        private void SetRandomCamRotation()
        {
            float xRot = Random.Range(-appData.CamRotJitter, appData.CamRotJitter);
            float yRot = Random.Range(-appData.CamRotJitter, appData.CamRotJitter);
            float zRot = Random.Range(-appData.CamRotJitter, appData.CamRotJitter);
            camera.transform.rotation = Quaternion.Euler(xRot, yRot, zRot);
        }


        private void Start()
        {
            
            environment = GameObject.Find("Environment").GetComponent<Environment>();
            appData = GameObject.Find("GameManager").GetComponent<GameManager>().appData;
            camera = Camera.main;
            m_Rigidbody = GetComponent<Rigidbody>();
            // controller = GetComponent<CharacterController>();

            nav_agent = GetComponent<NavMeshAgent>();

            props = GameObject.Find("3D MODELS/PROPS");
            player = GameObject.Find("Player");
            objectList = props.GetComponentsInChildren<Transform>();
            player_collider = player.GetComponent<Collider>();
            Bounds player_bounds = player_collider.bounds;
            player_size = player_bounds.size;
            // true if target is an existing object
            // false if target is a point in any floor that does not overlap with existing objects
            target_as_object = false; //doesnt work with shortest distancetype yet!

            // distance_type = "euclidian";
            // distance_type = "manhattan";
            distance_type = "shortest";
            // print("distance type " + distance_type + " target_as_object=" +target_as_object);
            // distance_type = "shortestpath"; //from navmesh, a* algotihm based but rotation does not depend on agent capabilities as nodes in path placed in polygons of the mesh and only the shortest path between nodes is found

            if (target_as_object)
            {
                // TO SELECT AN EXISTING OBJECT AS TARGET
                foreach (Transform obj in objectList) 
                {
                    if (obj.GetComponent<Collider>()!=null)
                    {
                        targets_list.Add(obj);

                        // print("objsname"+obj.name);
                        
                    }
                }
            }
            else
            {
                //TO CREATE A TARGET WITHIN A FLOOR

                floor_list = GameObject.FindGameObjectsWithTag("floor");
                
            }
        }

        public void Reset(int action)
        // can make conditional, if action is zero do validation vs training
        // SET AGENT POSITION
        {
            // THE ACTIONS ARE THE CONDITIONS (TRAINING CONDITION/VALIDATION CONDITION) IN THE TRAINING FILE
            if (action==0) //fixed starting position for agent in the middle
            {
                m_Rigidbody.position = new Vector3(0f, 1.1f, 0f);
                transform.rotation = Quaternion.Euler(0, 0, 0);
                agent_position1=m_Rigidbody.position;
            }

            if (action==1)  //random starting position for agent within one of the floors
            {
                // choose a floor 
                chosen_floor = Random.Range(0,floor_list.Length);
                floor = floor_list[chosen_floor];

                Mesh floor_mesh=floor.GetComponent<MeshFilter>().mesh;
                floor_collider= floor.GetComponent<Collider>();
                Bounds floor_bounds=floor_collider.bounds;
                floor_center=floor_bounds.center;
                floor_size=floor_bounds.size;



                
                agent_location = new Vector3(Random.Range(floor_center.x -  floor_size.x * 0.5f, floor_center.x +  floor_size.x  * 0.5f),
                        player_collider.bounds.center.y*0.5f,
                        Random.Range(floor_center.z -  floor_size.z * 0.5f, floor_center.z +  floor_size.z  * 0.5f));

                // check for colliders at the generated agent location 
                bool test_loc;
                test_loc = Physics.CheckSphere(agent_location, player_size.x*0.5f);
                // regenerate target location until it has no collision with existing objects to avoid overlaps
                while (test_loc==true)
                    {   
                        
                        agent_location = new Vector3(Random.Range(floor_center.x -  floor_size.x * 0.5f, floor_center.x +  floor_size.x  * 0.5f),
                            player_collider.bounds.center.y*0.5f,
                            Random.Range(floor_center.z -  floor_size.z * 0.5f, floor_center.z +  floor_size.z  * 0.5f));

                        test_loc = Physics.CheckSphere(agent_location, player_size.x*0.5f);
                    }

                m_Rigidbody.position = agent_location;
                transform.rotation = Quaternion.Euler(0, 0, 0);
                agent_position1=m_Rigidbody.position;
            }

            if (action==2)
            {
                m_Rigidbody.position = new Vector3(0f, 1.1f, 0f);
                transform.rotation = Quaternion.Euler(0, 0, 0);
                agent_position1=m_Rigidbody.position;
                validation=true;
            }
            if (action==3)
            {
                m_Rigidbody.position = new Vector3(0f, 1.1f, 0f);
                transform.rotation = Quaternion.Euler(0, 0, 0);
                agent_position1=m_Rigidbody.position;

                validation =false;
                
                episode=episode+1;
                Debug.Log("Episode"+episode);
            }
            _collidedWith = "";
            _stepCount = 0;
            // SET TARGET FOR AGENT
            UnityEngine.Object.DestroyImmediate(target_object);

            // create new target object upon reset
            if (target_as_object) //when target is an existing object
            {
                target_object = GameObject.CreatePrimitive(PrimitiveType.Sphere);

                MeshRenderer rend;
                rend = target_object.GetComponent<MeshRenderer>();
                rend.enabled = false; //MAKE FALSE WHEN TRAINING! to make target sphere invisible to agent

                int chosen_target = Random.Range(0,targets_list.Count);
                // print("targetslen" + targets_list.Count);

                target = targets_list[chosen_target];
                object_collider = target.GetComponent<Collider>();
                Bounds object_bounds = object_collider.bounds;
                object_size = object_bounds.size;
                target_location = object_bounds.center;
                x_bound = object_size.x;
                z_bound = object_size.z;
                target_object.transform.position = target_location;
                target_object.transform.localScale = new Vector3(x_bound+player_size.x*0.5f, 0f, z_bound+player_size.z*0.5f);
                
                // print("Target Location=" + target.name + target_location +"path " +target.transform.root.name); 
            }

            else //when target is a random point within a floor
            {
                //COMMENT OUT THESE CONDITIONS IF YOU ARE NOT CHANGING THE AREA TARGET GENERATED. WITH THIS CONDITIONS YOU ARE CALCULATING THE SCALE YOU CHANGE THE AREA AND YOU SHOULD USE IT AS 1 IF YOU DON'T SCALE THE AREA
               if (validation == true)
                {
                    chosen_floor = Random.Range(0,floor_list.Length);
                    // print("floorslen" + floor_list.Length);
                    scale=1f;
                    // print("validation scale" + scale);
                    floor = floor_list[chosen_floor];
                }
            
                if (validation == false)
                {
                    int c = max_episodes/100;
                    // print("c"+c);
                    // max_episodes=max_episodes+100;
                    chosen_floor = Random.Range(0,1);
                    
                    scale=0.2f*c;
                    floor = floor_list[chosen_floor]; 

                    if (episode>max_episodes)
                    {
                        if (episode<500)
                        {
                            max_episodes=max_episodes+100;
                        }
                        
                    }

                    if (episode>500)
                    {
                        chosen_floor = Random.Range(0,floor_list.Length);
                        scale=1f;
                        floor = floor_list[chosen_floor];
                    }
                }

            // IF YOU'RE KEEPING THE SCALE STABLE AS 1 SO THE TARGET IS GENERATED RANDOMLY YOU SHOULD CHOOSE THE FLOORS LIKE BELOW

                // chosen_floor = Random.Range(0,floor_list.Length);
                // scale=1f;
                // floor = floor_list[chosen_floor];


                Mesh floor_mesh=floor.GetComponent<MeshFilter>().mesh;
                floor_collider= floor.GetComponent<Collider>();
                Bounds floor_bounds=floor_collider.bounds;
                floor_center=floor_bounds.center;
                floor_size=floor_bounds.size;
                floor_size=scale*floor_size;
                // print("scale training="+ scale);

                // chosen_floor = Random.Range(0,floor_list.Length);
                // //         // scale=1f;
                // floor = floor_list[chosen_floor];
                // Mesh floor_mesh=floor.GetComponent<MeshFilter>().mesh;
                // floor_collider= floor.GetComponent<Collider>();
                // Bounds floor_bounds=floor_collider.bounds;
                // floor_center=floor_bounds.center;
                // floor_size=floor_bounds.size;
                // floor_size=floor_size;
                // floor_size= floor_size*scale;
                // generate random target location within selected floor
                target_location = new Vector3(Random.Range(floor_center.x -  floor_size.x * 0.5f, floor_center.x +  floor_size.x  * 0.5f),
                        player_collider.bounds.center.y*0.5f,
                        Random.Range(floor_center.z -  floor_size.z * 0.5f, floor_center.z +  floor_size.z  * 0.5f));
                bool test_box;
                test_box = Physics.CheckSphere(target_location, player_size.x*0.5f);
                float i_dist= Vector3.Distance(target_location, agent_location);  

                //Check if it can generate path to that target
                path = new NavMeshPath();

                bool path_existence;
                path_existence = NavMesh.CalculatePath(m_Rigidbody.position, target_location, NavMesh.AllAreas, path);
                // print("path status initial "+  path.status);

                if (path.status != NavMeshPathStatus.PathComplete)
                {
                    test_box=true;
                }

                if (Mathf.Abs(i_dist)<1)
                { 
                    test_box=true;
                }//check if it has no collision with existing objects to avoid overlaps
                
                while (test_box==true)
                {   
                    target_location = new Vector3(Random.Range(floor_center.x -  floor_size.x * 0.5f, floor_center.x +  floor_size.x  * 0.5f),
                        player_collider.bounds.center.y*0.5f,
                        Random.Range(floor_center.z -  floor_size.z * 0.5f, floor_center.z +  floor_size.z  * 0.5f));


                    test_box = Physics.CheckSphere(target_location, player_size.x*0.5f);
                    i_dist= Vector3.Distance(target_location, agent_location); 

                    if (Mathf.Abs(i_dist)<1)
                    { 
                        test_box=true;
                    }
                    path_existence = NavMesh.CalculatePath(m_Rigidbody.position, target_location, NavMesh.AllAreas, path);

                    if (path.status != NavMeshPathStatus.PathComplete)
                    {
                        test_box=true;
                    }
                }
                
                // create target object at the target location
                target_object = GameObject.CreatePrimitive(PrimitiveType.Cube);
                target_location.y=0.1f;
                target_object.transform.position = target_location;  
                target_object.transform.localScale =new Vector3(player_size.x*0.5f, 0f, player_size.z*0.5f);
                target_object.GetComponent<BoxCollider>().enabled = false;
                MeshRenderer rend;
                rend = target_object.GetComponent<MeshRenderer>();
                rend.enabled = false; //MAKE FALSE WHEN TRAINING! to make target cube invisible to agent

            }
            float initialdistancetotarget;
            initialdistancetotarget=0;

            if (distance_type == "euclidian")
            {
                Vector3 distance_xz;
                distance_xz = target_location- m_Rigidbody.position;
                distance_xz.y=0;
                initialdistancetotarget = distance_xz.magnitude;
            }
            
            if (distance_type == "manhattan")
            {
                initialdistancetotarget = Mathf.Abs(target_location.x-m_Rigidbody.position.x) + Mathf.Abs(target_location.z-m_Rigidbody.position.z);
            }

            if (distance_type == "shortest")
            {
                path = new NavMeshPath();
                // path.ClearCorners();

                bool path_existence;
                path_existence = NavMesh.CalculatePath(m_Rigidbody.position, target_location, NavMesh.AllAreas, path);

                while (path.status != NavMeshPathStatus.PathComplete)
                { 
                    path = new NavMeshPath();
                    
                    path_existence = NavMesh.CalculatePath(m_Rigidbody.position, target_location, NavMesh.AllAreas, path);
                }
                
                float lng = 0.0f;

                for ( int i = 0; i < path.corners.Length - 1; i++ )
                {
                    lng += Vector3.Distance(path.corners[i], path.corners[i + 1]);
                    // comment when running!
                    // Debug.DrawLine(path.corners[i], path.corners[i + 1], Color.red, 5f, false);
                }
                initialdistancetotarget = lng;

            }
            
            distance_afterstep = initialdistancetotarget;

            initialdistancetotarget= 10 * initialdistancetotarget;
            initialdistancetotarget=Mathf.Round(initialdistancetotarget);
            environment.DistancetoTarget = (byte)initialdistancetotarget; 
            _maxSteps = appData.MaxSteps;

        }

    }

}