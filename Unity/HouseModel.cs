using System.Collections.Generic;
using UnityEngine;
using System.Linq;
using ImgSynthesis = indoorMobility.Scripts.ImageSynthesis.ImgSynthesis;
using indoorMobility.Scripts.Utils;


namespace indoorMobility.Scripts.Game
{
    public class HouseModel : MonoBehaviour {
        #region;

        private AppData appData;
   
        #endregion;


        #region;

        public void Reset(int action)
        { 
        }

        private void Start() {
            appData = GameObject.Find("GameManager").GetComponent<GameManager>().appData;
            
        }
        #endregion;


    }
}