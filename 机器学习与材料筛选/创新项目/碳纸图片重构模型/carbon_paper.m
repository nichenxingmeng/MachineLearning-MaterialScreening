clear all;
clc;
Porosity = 0.6;   % Porosity
d = 3;            % Diameter
L = 400;           % Characteristic lenght
W = 400; 
s = 400;           % Thickness
K = 1;          % Isotropy coefficient.各向同性系数 From 0 (Full anisotropy) to 1 (Full isotropy)
% AUXILIARY VARIABLES CALCULATION

%r = d/2;                         % Fiber radius
PP = zeros(L,W,s);         % Output 3D binary image;输出3D二进制图像
E = 1;                           % Initial Porosity
[x,y,z] = meshgrid(0:W-1,0:L-1,0:s-1); % Meshing of the model网格化模型
figure
% hwait = waitbar(0,'Loading 0%'); % Waiting Bar

% GENERATION OF THE MATERIAL
r=d/2; 

box on;
axis ([0,W,0,L,0,s]);
view (3);
daspect([1 1 1]);
camlight;
lighting phong;
material dull;
while E>Porosity % Create fibers until the desired porosity is achieved

    str = ['Loading ',num2str((1-E)/(1-Porosity)*100),'%'];
%     waitbar((1-E)/(1-Porosity),hwait,str);
   
    % GENERATION OF FIBER PARAMETERS 
    % Orientation angles
    fi = rand(1)*2*pi;
    theta = acos(K*cos(rand(1)*pi));
    % Director vector
    u1 = sin(theta)*cos(fi); %
    u2 = sin(theta)*sin(fi); %
    u3 = cos(theta);
    % Orthonormal base of the normal plane to the director
    % vector正交基准的导向矢量的法向平面
    M = [sin(fi)     cos(theta)*cos(fi);
        -cos(fi)     cos(theta)*sin(fi);
           0            -sin(theta)   ];
    % Random point on the vectorial space 矢量空间上的随机点
    a=(L/2)*sqrt(2)*(2*rand(1)-1);
    b=(L/2)*sqrt(3)*(2*rand(1)-1);
    P0 = [L/2 L/2 s/2]' + M*[a b]';
    x0 = P0(1);
    y0 = P0(2);
    z0 = P0(3);
    
    % FIBER RENDERING/STORAGE光纤渲染/存储
    % Calculate distance-to-the-axis field   计算点（以新随机点为原点）到导向向量的距离场
    dist = sqrt( ( (z-z0)*u2-(y-y0)*u3 ).^2 + ( (x-x0)*u3-(z-z0)*u1 ).^2 + ( (y-y0)*u1-(x-x0)*u2 ).^2 );
    index = find(dist<=r);
    PP(index) = 1;
    % Render the surface渲染曲面
    patch(isosurface(x,y,z,dist,r),'facecolor','r','edgecolor','none');
    patch(isocaps(x,y,z,dist,r,'below'),'facecolor','r','edgecolor','none');
    % Update porosity value更新孔隙度值
    E = 1 - sum(PP(:))/((s+1)*(L+1)*(L+1)); 
end