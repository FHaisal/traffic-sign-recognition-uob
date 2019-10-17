function call_im(npic, folders)
global height width noc data target;

sBasePath = 'Images';
height = 32;
width = 32;
data = [];
target = [];
folders = folders - 1;
noc = folders + 1;

% 0:42
for nNumFolder = 0:folders
    sFolder = num2str(nNumFolder, '%05d');
    
    sPath = [sBasePath, '\', sFolder, '\'];

    if isfolder(sPath)
        [ImgFiles, Rois, Classes] = readSignData([sPath, '\GT-', num2str(nNumFolder, '%05d'), '.csv']);
        
        if nargin < 1
            npic = numel(ImgFiles);
        end
        
        nf = npic;
        if npic > numel(ImgFiles)
            nf = numel(ImgFiles);
        end
        
        for i = 1:nf
            ImgFile = [sPath, '\', ImgFiles{i}];
            Img = imread(ImgFile);
            
            %fprintf(1, 'Currently training: %s Class: %d Sample: %d / %d\n', ImgFiles{i}, Classes(i), i, numel(ImgFiles));

            Img = Img(Rois(i, 2) + 1:Rois(i, 4) + 1, Rois(i, 1) + 1:Rois(i, 3) + 1);
            Img_Resize = imresize(Img, [width, height]);

            MyTrainingFunction(Img_Resize, Classes(i));
        end
        save im data target noc;
    end
end

function [rImgFiles, rRois, rClasses] = readSignData(aFile)
% Reads the traffic sign data.
%
% aFile         Text file that contains the data for the traffic signs
%
% rImgFiles     Cell-Array (1 x n) of Strings containing the names of the image
%               files to operate on
% rRois         (n x 4)-Array containing upper left column, upper left row,
%               lower left column, lower left row of the region of interest
%               of the traffic sign image. The image itself can have a
%               small border so this data will give you the exact bounding
%               box of the sign in the image
% rClasses      (n x 1)-Array providing the classes for each traffic sign

    fID = fopen(aFile, 'r');
    
    % Discards the first header column in the .csv file
    fgetl(fID);
    
    f = textscan(fID, '%s %*d %*d %d %d %d %d %d', 'Delimiter', ';');
    
    rImgFiles = f{1}; 
    rRois = [f{2}, f{3}, f{4}, f{5}];
    rClasses = f{6};
    
    fclose(fID);
return

function MyTrainingFunction(img, class)
    global target data;
    %data = [data, reshape(img, 1024, 1)];
    data = [data, single(img(:))];
    target(numel(target) + 1) = class + 1;
return
