# Object-Detection 
# A simple project using VGG16 network to recognize objects

1. Download Data: https://drive.google.com/file/d/1pvprAYqRhAihJJ9qRU3fSLjnJ0POzMDb/view?usp=sharing
2. Download Model: https://drive.google.com/file/d/105aLZotxdU66OqknqmO5wdg3unebc446/view?usp=sharing
  
3. In the data there are already 5 objects, in case you want to add objects, follow these steps:
  * Step 1: Open make_data.py -> change object_label = "Name_of_object" at line 6 -> run make_data.py and show the object you want to the camera.
  * Step 2: Open train.py -> Uncomment #save_data(raw_folder) to make train_file.data (This command only run once, comment it when everything done !)
  * Step 3: In train.py, change number of classifier base on number of objects in Data folder: this number ![image](https://user-images.githubusercontent.com/85300544/120814540-2eb10580-c579-11eb-8a77-560e2d45c40c.png)
  * Step 4: Train model: Run train.py on device or GoogleColab -> Download weight and model.
  * Step 5: Open run.py -> Change class_name = ['Obj1', 'Obj2', 'Obj3', 'Obj4', "Obj5", ....] base on file name in Data folder (Remember these object's names must be in order!)
  Example: 
  
  
  
  ![image](https://user-images.githubusercontent.com/85300544/120816033-94ea5800-c57a-11eb-8bef-5944eeb2cd5a.png)  ![image](https://user-images.githubusercontent.com/85300544/120816066-9e73c000-c57a-11eb-942a-f050e265fe79.png)
  
  * Step 6: Run file run.py and see result  :D 



![image](https://user-images.githubusercontent.com/85300544/120816781-46898900-c57b-11eb-944a-4c5b2e85e643.png)


![image](https://user-images.githubusercontent.com/85300544/120817573-02e34f00-c57c-11eb-8795-8c6a9275d2aa.png)

![image](https://user-images.githubusercontent.com/85300544/120817816-3920ce80-c57c-11eb-9c18-e674be04f627.png)





 
