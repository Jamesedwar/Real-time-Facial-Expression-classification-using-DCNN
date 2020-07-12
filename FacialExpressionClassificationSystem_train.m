
function FacialExpressionClassificationSystem_train()
trn={'C:\FEB 2020\CNN\Code\train\angry\angry1.jpg', ...
'C:\FEB 2020\CNN\Code\train\angry\angry2.jpg', ...
'C:\FEB 2020\CNN\Code\train\angry\angry3.jpg',  ...
'C:\FEB 2020\CNN\Code\train\angry\angry4.jpg',  ...
'C:\FEB 2020\CNN\Code\train\angry\angry5.jpg',  ...
'C:\FEB 2020\CNN\Code\train\angry\angry6.jpg',  ...
'C:\FEB 2020\CNN\Code\train\angry\angry7.jpg',   ...
'C:\FEB 2020\CNN\Code\train\angry\angry8.jpg',   ...
'C:\FEB 2020\CNN\Code\train\angry\angry9.jpg',  ...
'C:\FEB 2020\CNN\Code\train\angry\angry10.jpg',  ...
'C:\FEB 2020\CNN\Code\train\disgust\disgust1.jpg',  ...
'C:\FEB 2020\CNN\Code\train\disgust\disgust2.jpg',  ...
'C:\FEB 2020\CNN\Code\train\disgust\disgust3.jpg',  ...
'C:\FEB 2020\CNN\Code\train\disgust\disgust4.jpg',  ...
'C:\FEB 2020\CNN\Code\train\disgust\disgust5.jpg',  ...
'C:\FEB 2020\CNN\Code\train\disgust\disgust6.jpg',  ...
'C:\FEB 2020\CNN\Code\train\disgust\disgust7.jpg',  ...
'C:\FEB 2020\CNN\Code\train\disgust\disgust8.jpg',  ... 
'C:\FEB 2020\CNN\Code\train\disgust\disgust9.jpg',  ...
'C:\FEB 2020\CNN\Code\train\disgust\disgust10.jpg',  ...
'C:\FEB 2020\CNN\Code\train\fear\fear1.jpg',  ...
'C:\FEB 2020\CNN\Code\train\fear\fear2.jpg',  ...
'C:\FEB 2020\CNN\Code\train\fear\fear3.jpg',  ...
'C:\FEB 2020\CNN\Code\train\fear\fear4.jpg',  ...
'C:\FEB 2020\CNN\Code\train\fear\fear5.jpg',  ...
'C:\FEB 2020\CNN\Code\train\fear\fear6.jpg',  ...
'C:\FEB 2020\CNN\Code\train\fear\fear7.jpg',  ...
'C:\FEB 2020\CNN\Code\train\fear\fear8.jpg',  ...
'C:\FEB 2020\CNN\Code\train\fear\fear9.jpg',  ...
'C:\FEB 2020\CNN\Code\train\fear\fear10.jpg',  ...
'C:\FEB 2020\CNN\Code\train\happy\happy1.jpg',  ...
'C:\FEB 2020\CNN\Code\train\happy\happy2.jpg',  ...
'C:\FEB 2020\CNN\Code\train\happy\happy3.jpg',  ...
'C:\FEB 2020\CNN\Code\train\happy\happy4.jpg',  ...
'C:\FEB 2020\CNN\Code\train\happy\happy5.jpg',  ...
'C:\FEB 2020\CNN\Code\train\happy\happy6.jpg',  ...
'C:\FEB 2020\CNN\Code\train\happy\happy7.jpg',  ...
'C:\FEB 2020\CNN\Code\train\happy\happy8.jpg',  ...
'C:\FEB 2020\CNN\Code\train\happy\happy9.jpg',  ...
'C:\FEB 2020\CNN\Code\train\happy\happy10.jpg',  ...
'C:\FEB 2020\CNN\Code\train\neutral\neutral1.jpg',  ...
'C:\FEB 2020\CNN\Code\train\neutral\neutral2.jpg',  ...
'C:\FEB 2020\CNN\Code\train\neutral\neutral3.jpg',  ...
'C:\FEB 2020\CNN\Code\train\neutral\neutral4.jpg',  ...
'C:\FEB 2020\CNN\Code\train\neutral\neutral5.jpg',  ...
'C:\FEB 2020\CNN\Code\train\neutral\neutral6.jpg',  ...
'C:\FEB 2020\CNN\Code\train\neutral\neutral7.jpg',  ...
'C:\FEB 2020\CNN\Code\train\neutral\neutral8.jpg',  ...
'C:\FEB 2020\CNN\Code\train\neutral\neutral9.jpg',  ...
'C:\FEB 2020\CNN\Code\train\neutral\neutral10.jpg',  ...
'C:\FEB 2020\CNN\Code\train\sad\sad1.jpg',  ...
'C:\FEB 2020\CNN\Code\train\sad\sad2.jpg',  ...
'C:\FEB 2020\CNN\Code\train\sad\sad3.jpg',  ...
'C:\FEB 2020\CNN\Code\train\sad\sad4.jpg',  ...
'C:\FEB 2020\CNN\Code\train\sad\sad5.jpg',  ...
'C:\FEB 2020\CNN\Code\train\sad\sad6.jpg',  ...
'C:\FEB 2020\CNN\Code\train\sad\sad7.jpg',  ...
'C:\FEB 2020\CNN\Code\train\sad\sad8.jpg',  ...
'C:\FEB 2020\CNN\Code\train\sad\sad9.jpg',  ...
'C:\FEB 2020\CNN\Code\train\sad\sad10.jpg',  ...
'C:\FEB 2020\CNN\Code\train\surprise\surprise1.jpg',  ...
'C:\FEB 2020\CNN\Code\train\surprise\surprise2.jpg',  ...
'C:\FEB 2020\CNN\Code\train\surprise\surprise3.jpg',  ...
'C:\FEB 2020\CNN\Code\train\surprise\surprise4.jpg',  ...
'C:\FEB 2020\CNN\Code\train\surprise\surprise5.jpg',  ...
'C:\FEB 2020\CNN\Code\train\surprise\surprise6.jpg',  ...
'C:\FEB 2020\CNN\Code\train\surprise\surprise7.jpg',  ...
'C:\FEB 2020\CNN\Code\train\surprise\surprise8.jpg',  ...
'C:\FEB 2020\CNN\Code\train\surprise\surprise9.jpg',  ...
'C:\FEB 2020\CNN\Code\train\surprise\surprise10.jpg'};  

F2=0;
for t=1:70

img = imread(trn{t}); 
% img =  rgb2gray(img1);
FaceDetect = vision.CascadeObjectDetector; 
BB = step(FaceDetect, img); 
 
for i = 1 : size(BB,1)     
  rectangle('Position', BB(i,:), 'LineWidth', 3, 'LineStyle', '-', 'EdgeColor', 'r'); 
end 
for i = 1 : size(BB, 1) 
  J = imcrop(img, BB(i, :)); 
   
  F2 = imresize(J,[50 50]);
  
end
imwrite(F2,trn{t});
end
disp('Success');
end