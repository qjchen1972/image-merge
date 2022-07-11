A simple program based on  paper<< Object recognition from local scale-invariant features>> , The project provides some features:
* Image merge 
* image similarity 

![](https://github.com/qjchen1972/image-merge/blob/main/img/sim.png)

In the image above, the left side is the full image, and the image on the right side is used to identify the area in the full image to accurately find out the differences

Install
===
* pip install -r requirements.txt

Getting Started
====
* image merge
  
  python imgmerge -m 0
  
* Merge images under directory
  
  python imgmerge -m 2
  
* image  similarity  
  
  python imgmerge -m 1
