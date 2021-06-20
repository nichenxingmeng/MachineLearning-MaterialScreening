const float ps=2.16;
float mx1,mx2=0.0,mxin,mxout,my1=0.0,my2=0.0,mz1=0.0,mxx=1.0,mx5=1.0,mave;	
char permeabilityPath[30]="Permeability/",PermeabilityPath[60];
void calculate_permeability()
					{
						int x,y,z;
						float rho,vx,vy,vz,tor;
						mxx=mx5;
					    mx1=0.0;mxin=0.0;mxout=0.0;my1=0.0;my2=0.0;mz1=0.0;
						
						for (z=0;z<Nz;z++)
						{
							for (y=0;y<Ny;y++)
							{
								for (x=0;x<Nx;x++)
								{
									if (flag[z][y][x]==1)
										continue;
									else
									{
									rho=f0[z][y][x]+f1[z][y][x]+f2[z][y][x]+f3[z][y][x]+f4[z][y][x]+
										     f5[z][y][x]+f6[z][y][x]+f7[z][y][x]+f8[z][y][x]+f9[z][y][x]+
										     f10[z][y][x]+f11[z][y][x]+f12[z][y][x]+f13[z][y][x]+f14[z][y][x]+
										     f15[z][y][x]+f16[z][y][x]+f17[z][y][x]+f18[z][y][x];
									
										vx=(f1[z][y][x]-f2[z][y][x]+f7[z][y][x]+f8[z][y][x]-f9[z][y][x]-
											f10[z][y][x]+f15[z][y][x]+f18[z][y][x]-f16[z][y][x]-f17[z][y][x]);//ÅÐ¶ÏÊÇ·ñ³ýÃÜ¶È
										vy=(f3[z][y][x]-f4[z][y][x]+f12[z][y][x]+f13[z][y][x]-f11[z][y][x]-
											f14[z][y][x]+f8[z][y][x]+f9[z][y][x]-f7[z][y][x]-f10[z][y][x]);
										vz=(f5[z][y][x]-f6[z][y][x]+f15[z][y][x]+f16[z][y][x]-f17[z][y][x]-
											f18[z][y][x]+f11[z][y][x]+f12[z][y][x]-f13[z][y][x]-f14[z][y][x]);

										mx1=mx1+vx;
										my1=my1+vy;
										mz1=mz1+vz;
									}
								}
							}
						}

						    
							//mave=sqrt(mx1*mx1+my1*my1+mz1*mz1);
							//tor=mave/mz1;//flow in z
							//tor=mave/my1;//flow in y
							//tor=mave/mx1;//flow in x

						//....................................................................
							mx5=mz1;
							mxx=(mz1-mxx)/mz1;
							if(mxx<0) 
							{
								mxx=-mxx;
							}
							

							float u=0.001;                   // mm**2/s 
							float dz=ps/1000;//dx=ps/(1000.0*NE); // mm
							float dt=dz*dz*(tau-0.5)/(3*u);
							float visco=(tau-0.5)/3.0;

							mx1=(3*Nz*mx1)/(Nx*Ny*Nz)/(rho_1-rho_0);
							my1=(3*Nz*my1/(Nx*Ny*Nz))/(rho_1-rho_0);
							mz1=(3*Nz*mz1/(Nx*Ny*Nz))/(rho_1-rho_0);

							mx1=rho*mx1*dz*dz*visco;//*60*60*24*g1/u/100;
							my1=rho*my1*dz*dz*visco;//*60*60*24*g1/u/100;
							mz1=rho*mz1*dz*dz*visco;//*60*60*24*g1/u/100;

							//tor=powf(porosity,0.5);
							//tor=1/tor;


							cout.precision(7);
							cout.setf(ios::showpoint);cout.setf(ios::fixed);
							//cout<<"Time="<<t<<" Error="<<mxx<<"  |  kxx="<<mx1<<"(mm^2)    kxy="<<my1<<"(mm^2)   kxz="<<mz1<<"(mm^2)"<<"tortuosity="<<tor<<endl; 
							cout<<"Time="<<t<<" Error="<<mxx<<"  |  kxx="<<mx1<<"(mm^2)    kxy="<<my1<<"(mm^2)   kxz="<<mz1<<"(mm^2)"<<endl; 
							char tname1[30],tname2[30];
							int T=t/permeability_interval;
							ofstream permeabilityflow;
							stringstream tstring;
							tstring<<T;
							tstring>>tname2;
							strcpy(tname1,tname2);
							strcpy(PermeabilityPath,permeabilityPath);
							strcat(strcat(PermeabilityPath,tname1),".dat");
							permeabilityflow.open(PermeabilityPath,ios::trunc);

							permeabilityflow<<"Time="<<t<<" Error="<<mxx<<"  |  kxx="<<mx1<<"(mm^2)    kxy="<<my1<<"(mm^2)   kxz="<<mz1<<"(mm^2)"<<"   porosity ="<<porosity;
							permeabilityflow.close();

							
					}
