#include "molecule.h"
#include "memory.h"
#include "util.h"
#include <cstring>
#include <iostream>

MOLECULE::MOLECULE():
n_atom(0),n_bond(0),n_angle(0),n_dihedral(0),
bond(nullptr), angle(nullptr), dihedral(nullptr),
// bond_coef(nullptr), angle_coef(nullptr), dihedral_coef(nullptr),
special(nullptr)
{}

MOLECULE::~MOLECULE()
{
    if(bond) destroy2DArray(bond);
    if(angle) destroy2DArray(angle);
    if(dihedral) destroy2DArray(dihedral);
    // if(bond_coef) destroy2DArray(bond_coef);
    // if(angle_coef) destroy2DArray(angle_coef);
    // if(dihedral_coef) destroy2DArray(dihedral_coef);
    if(special) destroy2DArray(special);
}

void MOLECULE::from_file(const char *fname)
{
    FILE *fp=fopen(fname,"r");
    char buf[SZ_FBF],buf2[SZ_FBF],*_buf;
    enum {MFS_HEADER,MFS_BOND,MFS_ANGLE,MFS_DIHEDRAL,MFS_OTHER} status=MFS_HEADER;
    int temp,temp2;
    std::cerr<<"reading "<<fname<<"\n";
    int nb=0,na=0,nd=0,*dattp;
    while(true)
    {
        fgets(buf,SZ_FBF,fp);
        if(feof(fp)) break;
        if((_buf=non_empty_string(buf)))
        {
            // std::cerr<<_buf;
            if(strncmp(_buf,"Coords",6)==0) status=MFS_OTHER;
            else if(strncmp(_buf,"Types",5)==0) status=MFS_OTHER;
            else if(strncmp(_buf,"Charges",7)==0) status=MFS_OTHER;
            else if(strncmp(_buf,"Bonds",5)==0) status=MFS_BOND;
            else if(strncmp(_buf,"Angles",6)==0) status=MFS_ANGLE;
            else if(strncmp(_buf,"Dihedrals",9)==0) status=MFS_DIHEDRAL;
            else
            {
                switch (status)
                {
                case MFS_HEADER:
                    if(sscanf(_buf,"%d %s",&temp,buf2) != 2) END_PROGRAM("read error");
                    if(strncmp(buf2,"atoms",5)==0) 
                    {
                        n_atom=temp;
                        create2DArray(special,n_atom,n_atom);
                    }
                    else if(strncmp(buf2,"bonds",5)==0)
                    {
                        n_bond=temp;
                        create2DArray(bond,n_bond,3);
                    }
                    else if(strncmp(buf2,"angles",6)==0)
                    {
                        n_angle=temp;
                        create2DArray(angle,n_angle,4);
                    }
                    else if(strncmp(buf2,"dihedrals",9)==0)
                    {
                        n_dihedral=temp;
                        create2DArray(dihedral,n_dihedral,5);
                    }
                    else END_PROGRAM("read error");
                    break;
                case MFS_BOND:
                    // std::cerr<<"bond "<<nb<<"\n";
                    dattp=bond[nb++];
                    if(sscanf(_buf,"%*d %d %d %d",dattp,dattp+1,dattp+2) != 3) END_PROGRAM("read error");
                    break;
                case MFS_ANGLE:
                    // std::cerr<<"angle "<<na<<"\n";
                    dattp=angle[na++];
                    if(sscanf(_buf,"%*d %d %d %d %d",dattp,dattp+1,dattp+2,dattp+3) != 4) END_PROGRAM("read error");
                    break;
                case MFS_DIHEDRAL:
                    // std::cerr<<"dihedral "<<nd<<"\n";
                    dattp=dihedral[nd++];
                    if(sscanf(_buf,"%*d %d %d %d %d %d",dattp,dattp+1,dattp+2,dattp+3,dattp+4) != 5) END_PROGRAM("read error");
                    break;
                default:
                    // default is do nothing
                    break;
                }
            }
        }
    }
    fclose(fp);
    std::cerr<<n_atom<<" atoms; "<<n_bond<<" bonds; "<<n_angle<<" angles; "<<n_dihedral<<" dihedrals\n";
    int *spc;
    for(int i=0;i<n_atom;++i)
    {
        spc=special[i];
        for(int j=0;j<n_atom;++j)
            spc[j]=0;
    }
    for(int i=0;i<n_bond;++i)
    {
        dattp=bond[i];
        --dattp[0];
        temp=--dattp[1];
        temp2=--dattp[2];
        special[temp][temp2]=special[temp2][temp]=1;
    }
    for(int i=0;i<n_angle;++i)
    {
        dattp=angle[i];
        --dattp[0];
        temp=--dattp[1];
        --dattp[2];
        temp2=--dattp[3];
        special[temp][temp2]=special[temp2][temp]=2;
    }
    for(int i=0;i<n_dihedral;++i)
    {
        dattp=dihedral[i];
        --dattp[0];
        temp=--dattp[1];
        --dattp[2];
        --dattp[3];
        temp2=--dattp[4];
        special[temp][temp2]=special[temp2][temp]=3;
    }
    // printMatrix(special_coul,2,2,"coul");

}
