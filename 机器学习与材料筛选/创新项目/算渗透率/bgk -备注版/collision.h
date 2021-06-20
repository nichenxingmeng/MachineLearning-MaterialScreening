#include "stdafx.h"

#include <iostream>

#include <fstream>
#include <sstream>
#include <cmath>
#include <float.h>
using namespace std;
float den;//porosity;
void collision()
{
	int x,y,z,i;
	float eps=0;
	float rho,vx,vy,vz;//density and velocity
// 	double	f_eq0[100][100][100],f_eq1[100][100][100],f_eq2[100][100][100],f_eq3[100][100][100],//equilibrium distribution
// 		f_eq4[100][100][100],f_eq5[100][100][100],f_eq6[100][100][100],f_eq7[100][100][100],
// 		f_eq8[100][100][100],f_eq9[100][100][100],f_eq10[100][100][100],f_eq11[100][100][100],
// 		f_eq12[100][100][100],f_eq13[100][100][100],f_eq14[100][100][100],f_eq15[100][100][100],
//		f_eq16[100][100][100],f_eq17[100][100][100],f_eq18[100][100][100];
	float	square,tau_inv,dummy,product;
	tau_inv=1.0/tau;
	for(z=0;z<Nz;z++)
	{
		for(y=0;y<Ny;y++)
		{
			for(x=0;x<Nx;x++)
			{
				if(flag[z][y][x]!=1 )// or if(flag[z][y][x]==0 )
				{
					//fluid node:compute\Omega_i(\rho,\vec v).
					//Omega_i depends on f_i^eq and f_i,
					//to compute f_i^eq you need density and velocities
					rho=f0[z][y][x]+f1[z][y][x]+f2[z][y][x]+f3[z][y][x]+f4[z][y][x]+
						f5[z][y][x]+f6[z][y][x]+f7[z][y][x]+f8[z][y][x]+f9[z][y][x]+
						f10[z][y][x]+f11[z][y][x]+f12[z][y][x]+f13[z][y][x]+f14[z][y][x]+
						f15[z][y][x]+f16[z][y][x]+f17[z][y][x]+f18[z][y][x];
					vx=(f1[z][y][x]-f2[z][y][x]+f7[z][y][x]+f8[z][y][x]-f9[z][y][x]-
						f10[z][y][x]+f15[z][y][x]+f18[z][y][x]-f16[z][y][x]-f17[z][y][x]);//去掉了除密度，想到底要不要除密度
					vy=(f3[z][y][x]-f4[z][y][x]+f12[z][y][x]+f13[z][y][x]-f11[z][y][x]-
						f14[z][y][x]+f8[z][y][x]+f9[z][y][x]-f7[z][y][x]-f10[z][y][x]);
					vz=(f5[z][y][x]-f6[z][y][x]+f15[z][y][x]+f16[z][y][x]-f17[z][y][x]-
						f18[z][y][x]+f11[z][y][x]+f12[z][y][x]-f13[z][y][x]-f14[z][y][x]);

					den=rho;

				
					//compute all f_i^eq from rho and v
					square=1.5*(vx*vx+vy*vy+vz*vz);
					f_eq0[z][y][x]=1.0/3.0*(den-square);
					rho=1.0/18.0;
					f_eq1[z][y][x]=rho*(den+3.0*vx+4.5*vx*vx-square);//将density写在了括号里，下面同，分析是不是应该这么写
					f_eq2[z][y][x]=f_eq1[z][y][x]-6.0*vx*rho;
					f_eq3[z][y][x]=rho*(den+3.0*vy+4.5*vy*vy-square);
					f_eq4[z][y][x]=f_eq3[z][y][x]-6.0*vy*rho;
					f_eq5[z][y][x]=rho*(den+3.0*vz+4.5*vz*vz-square);
					f_eq6[z][y][x]=f_eq5[z][y][x]-6.0*vz*rho;
				

					rho=1.0/36.0;
					product=vx+vy;
					f_eq8[z][y][x]=rho*(den+3.0*product+4.5*product*product-square);//den=1.0
					f_eq10[z][y][x]=f_eq8[z][y][x]-6.0*product*rho;
					product=-vx+vy;
					f_eq9[z][y][x]=rho*(den+3.0*product+4.5*product*product-square);
					f_eq7[z][y][x]=f_eq9[z][y][x]-6.0*product*rho;

					product=vy+vz;
					f_eq12[z][y][x]=rho*(den+3.0*product+4.5*product*product-square);
					f_eq14[z][y][x]=f_eq12[z][y][x]-6.0*product*rho;
					product=-vy+vz;
					f_eq11[z][y][x]=rho*(den+3.0*product+4.5*product*product-square);
					f_eq13[z][y][x]=f_eq11[z][y][x]-6.0*product*rho;

					product=vx+vz;
					f_eq15[z][y][x]=rho*(den+3.0*product+4.5*product*product-square);
					f_eq17[z][y][x]=f_eq15[z][y][x]-6.0*product*rho;
					product=-vx+vz;
					f_eq16[z][y][x]=rho*(den+3.0*product+4.5*product*product-square);
					f_eq18[z][y][x]=f_eq16[z][y][x]-6.0*product*rho;
					//modify distributions according to collision contribute the term after +=is \Omega_i
					f_eq0[z][y][x]=f0[z][y][x]+(f_eq0[z][y][x]-f0[z][y][x])/tau; //f0为上一个状态的分布函数，f_eq0为平衡分布函数
					f_eq1[z][y][x]=f1[z][y][x]+(f_eq1[z][y][x]-f1[z][y][x])/tau;
					f_eq2[z][y][x]=f2[z][y][x]+(f_eq2[z][y][x]-f2[z][y][x])/tau;
					f_eq3[z][y][x]=f3[z][y][x]+(f_eq3[z][y][x]-f3[z][y][x])/tau;
					f_eq4[z][y][x]=f4[z][y][x]+(f_eq4[z][y][x]-f4[z][y][x])/tau;
					f_eq5[z][y][x]=f5[z][y][x]+(f_eq5[z][y][x]-f5[z][y][x])/tau;
					f_eq6[z][y][x]=f6[z][y][x]+(f_eq6[z][y][x]-f6[z][y][x])/tau;
					f_eq7[z][y][x]=f7[z][y][x]+(f_eq7[z][y][x]-f7[z][y][x])/tau;
					f_eq8[z][y][x]=f8[z][y][x]+(f_eq8[z][y][x]-f8[z][y][x])/tau;
					f_eq9[z][y][x]=f9[z][y][x]+(f_eq9[z][y][x]-f9[z][y][x])/tau;
					f_eq10[z][y][x]=f10[z][y][x]+(f_eq10[z][y][x]-f10[z][y][x])/tau;
					f_eq11[z][y][x]=f11[z][y][x]+(f_eq11[z][y][x]-f11[z][y][x])/tau;
					f_eq12[z][y][x]=f12[z][y][x]+(f_eq12[z][y][x]-f12[z][y][x])/tau;
					f_eq13[z][y][x]=f13[z][y][x]+(f_eq13[z][y][x]-f13[z][y][x])/tau;
					f_eq14[z][y][x]=f14[z][y][x]+(f_eq14[z][y][x]-f14[z][y][x])/tau;
					f_eq15[z][y][x]=f15[z][y][x]+(f_eq15[z][y][x]-f15[z][y][x])/tau;
					f_eq16[z][y][x]=f16[z][y][x]+(f_eq16[z][y][x]-f16[z][y][x])/tau;
					f_eq17[z][y][x]=f17[z][y][x]+(f_eq17[z][y][x]-f17[z][y][x])/tau;
					f_eq18[z][y][x]=f18[z][y][x]+(f_eq18[z][y][x]-f18[z][y][x])/tau;
				}

		if (f_eq5[z][y][x]>5.0)
					{
					cout<<"f_eq5 >5.0 at location: "<< x  << y << z <<endl;
					}
				}
			}
		}
}