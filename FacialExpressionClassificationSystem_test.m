function FacialExpressionClassificationSystem_test()
tst={'C:\FEB 2020\CNN\Code\test\angry\angry1.jpg', ...
'C:\FEB 2020\CNN\Code\test\angry\angry2.jpg', ...
'C:\FEB 2020\CNN\Code\test\angry\angry3.jpg',  ...
'C:\FEB 2020\CNN\Code\test\angry\angry4.jpg',  ...
'C:\FEB 2020\CNN\Code\test\angry\angry5.jpg',  ...
'C:\FEB 2020\CNN\Code\test\angry\angry6.jpg',  ...
'C:\FEB 2020\CNN\Code\test\angry\angry7.jpg',   ...
'C:\FEB 2020\CNN\Code\test\angry\angry8.jpg',   ...
'C:\FEB 2020\CNN\Code\test\angry\angry9.jpg',  ...
'C:\FEB 2020\CNN\Code\test\angry\angry10.jpg',  ...
'C:\FEB 2020\CNN\Code\test\disgust\disgust1.jpg',  ...
'C:\FEB 2020\CNN\Code\test\disgust\disgust2.jpg',  ...
'C:\FEB 2020\CNN\Code\test\disgust\disgust3.jpg',  ...
'C:\FEB 2020\CNN\Code\test\disgust\disgust4.jpg',  ...
'C:\FEB 2020\CNN\Code\test\disgust\disgust5.jpg',  ...
'C:\FEB 2020\CNN\Code\test\disgust\disgust6.jpg',  ...
'C:\FEB 2020\CNN\Code\test\disgust\disgust7.jpg',  ...
'C:\FEB 2020\CNN\Code\test\disgust\disgust8.jpg',  ... 
'C:\FEB 2020\CNN\Code\test\disgust\disgust9.jpg',  ...
'C:\FEB 2020\CNN\Code\test\disgust\disgust10.jpg',  ...
'C:\FEB 2020\CNN\Code\test\fear\fear1.jpg',  ...
'C:\FEB 2020\CNN\Code\test\fear\fear2.jpg',  ...
'C:\FEB 2020\CNN\Code\test\fear\fear3.jpg',  ...
'C:\FEB 2020\CNN\Code\test\fear\fear4.jpg',  ...
'C:\FEB 2020\CNN\Code\test\fear\fear5.jpg',  ...
'C:\FEB 2020\CNN\Code\test\fear\fear6.jpg',  ...
'C:\FEB 2020\CNN\Code\test\fear\fear7.jpg',  ...
'C:\FEB 2020\CNN\Code\test\fear\fear8.jpg',  ...
'C:\FEB 2020\CNN\Code\test\fear\fear9.jpg',  ...
'C:\FEB 2020\CNN\Code\test\fear\fear10.jpg',  ...
'C:\FEB 2020\CNN\Code\test\happy\happy1.jpg',  ...
'C:\FEB 2020\CNN\Code\test\happy\happy2.jpg',  ...
'C:\FEB 2020\CNN\Code\test\happy\happy3.jpg',  ...
'C:\FEB 2020\CNN\Code\test\happy\happy4.jpg',  ...
'C:\FEB 2020\CNN\Code\test\happy\happy5.jpg',  ...
'C:\FEB 2020\CNN\Code\test\happy\happy6.jpg',  ...
'C:\FEB 2020\CNN\Code\test\happy\happy7.jpg',  ...
'C:\FEB 2020\CNN\Code\test\happy\happy8.jpg',  ...
'C:\FEB 2020\CNN\Code\test\happy\happy9.jpg',  ...
'C:\FEB 2020\CNN\Code\test\happy\happy10.jpg',  ...
'C:\FEB 2020\CNN\Code\test\neutral\neutral1.jpg',  ...
'C:\FEB 2020\CNN\Code\test\neutral\neutral2.jpg',  ...
'C:\FEB 2020\CNN\Code\test\neutral\neutral3.jpg',  ...
'C:\FEB 2020\CNN\Code\test\neutral\neutral4.jpg',  ...
'C:\FEB 2020\CNN\Code\test\neutral\neutral5.jpg',  ...
'C:\FEB 2020\CNN\Code\test\neutral\neutral6.jpg',  ...
'C:\FEB 2020\CNN\Code\test\neutral\neutral7.jpg',  ...
'C:\FEB 2020\CNN\Code\test\neutral\neutral8.jpg',  ...
'C:\FEB 2020\CNN\Code\test\neutral\neutral9.jpg',  ...
'C:\FEB 2020\CNN\Code\test\neutral\neutral10.jpg',  ...
'C:\FEB 2020\CNN\Code\test\sad\sad1.jpg',  ...
'C:\FEB 2020\CNN\Code\test\sad\sad2.jpg',  ...
'C:\FEB 2020\CNN\Code\test\sad\sad3.jpg',  ...
'C:\FEB 2020\CNN\Code\test\sad\sad4.jpg',  ...
'C:\FEB 2020\CNN\Code\test\sad\sad5.jpg',  ...
'C:\FEB 2020\CNN\Code\test\sad\sad6.jpg',  ...
'C:\FEB 2020\CNN\Code\test\sad\sad7.jpg',  ...
'C:\FEB 2020\CNN\Code\test\sad\sad8.jpg',  ...
'C:\FEB 2020\CNN\Code\test\sad\sad9.jpg',  ...
'C:\FEB 2020\CNN\Code\test\sad\sad10.jpg',  ...
'C:\FEB 2020\CNN\Code\test\surprise\surprise1.jpg',  ...
'C:\FEB 2020\CNN\Code\test\surprise\surprise2.jpg',  ...
'C:\FEB 2020\CNN\Code\test\surprise\surprise3.jpg',  ...
'C:\FEB 2020\CNN\Code\test\surprise\surprise4.jpg',  ...
'C:\FEB 2020\CNN\Code\test\surprise\surprise5.jpg',  ...
'C:\FEB 2020\CNN\Code\test\surprise\surprise6.jpg',  ...
'C:\FEB 2020\CNN\Code\test\surprise\surprise7.jpg',  ...
'C:\FEB 2020\CNN\Code\test\surprise\surprise8.jpg',  ...
'C:\FEB 2020\CNN\Code\test\surprise\surprise9.jpg',  ...
'C:\FEB 2020\CNN\Code\test\surprise\surprise10.jpg'};  
clc; 

for t=1:70

img = imread(tst{t}); 
% img =  rgb2gray(img1);
FaceDetect = vision.CascadeObjectDetector; 
FaceDetect.MergeThreshold = 7 ;
BB = step(FaceDetect, img); 

for i = 1 : size(BB,1)     
  rectangle('Position', BB(i,:), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'r'); 
end 
for i = 1 : size(BB, 1) 
  J = imcrop(img, BB(i, :)); 
    
  F2 = imresize(J,[50 50]);
  
end
imwrite(F2,tst{t});
end
disp('Success');

end