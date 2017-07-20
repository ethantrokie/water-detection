import  predictFunctions
import trainSVM

#predictFunctions.moduleC("/data/beehive/VideoWaterDatabase/videos/water/pond/pond_020.avi","/data/beehive/VideoWaterDatabase/masks/water/pond/pond_020.png",100,4,0,4,30,100,20,6)
#predictFunctions.predictFromOtherSVM("/data/Video.MOV","/data/beehive/ForTesting/masks/water/river/river_024.png",50,2,0,4,50,100,10,32)
predictFunctions.testFullVid("/data/Video.MOV","/data/beehive/ForTesting/masks/water/canal/canal_024.png","/data/beehive/Results/",50,2,0,5,50,100,10,32)
#trainSVM.main()