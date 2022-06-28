#include <cstring>
#include "useful.h"
#include <Eigen/Dense>

void ComputeOrientation(Vector3* inertia, Matrix3* orientation, Vector3* pos, Vector3* com, float* m, int2* groupID, float* scaleFactor, 
int numGrouped, int numRigid)
{
    using namespace Eigen;
    Matrix3f* I = new Matrix3f [numRigid];
    memset(I, 0, sizeof(Matrix3f)*numRigid);
    //compute moment here
    for(int idx = 0; idx < numGrouped; ++idx)
    {
        int2 i = groupID[idx];
        Vector3 dr = pos[i.y]-com[i.x];
        float mass = m[i.y]*scaleFactor[i.x];
        I[i.x](0,0) += (dr.y*dr.y+dr.z*dr.z)*mass;
        I[i.x](0,1) += -(dr.x*dr.y)*mass;
        I[i.x](1,0) = I[i.x](0,1);
        I[i.x](0,2) += -(dr.x*dr.z)*mass;
        I[i.x](2,0) = I[i.x](0,2);
        I[i.x](1,1) += (dr.x*dr.x+dr.z*dr.z)*mass;
        I[i.x](1,2) += -(dr.y*dr.z)*mass;
        I[i.x](2,1) = I[i.x](1,2);
        I[i.x](2,2) += (dr.x*dr.x+dr.y*dr.y)*mass;
    }
    //diagonize moment here by Eigen
    //Vector3* inertia = new Vector3[_numRigid];
    //Matrix3* orientation = new Matrix3[_numRigid];
    for(int i = 0; i < numRigid; ++i)
    {
        Matrix3f tmp = I[i];
        SelfAdjointEigenSolver<Matrix3f> solver(tmp);
        Vector3f lambda = solver.eigenvalues();
        Matrix3f M = solver.eigenvectors();
        //change to right-handed coordinates
        if (M.determinant() < 0.)
        {
            Vector3f v = M.col(2);
            M.col(2) = M.col(1);
            M.col(1) = v;
            float l = lambda(2);
            lambda(2) = lambda(1);
            lambda(1) = l;
        }
        inertia[i].x = lambda(0);
        inertia[i].y = lambda(1);
        inertia[i].z = lambda(2);

        orientation[i].exx = M(0,0);
        orientation[i].exy = M(0,1);
        orientation[i].exz = M(0,2);
        orientation[i].eyx = M(1,0);
        orientation[i].eyy = M(1,1);
        orientation[i].eyz = M(1,2);
        orientation[i].ezx = M(2,0);
        orientation[i].ezy = M(2,1);
        orientation[i].ezz = M(2,2);

        #ifdef debug
        std::ofstream infile("moment_inertia.txt");
        infile << inertia[i] << std::endl;
        infile << orientation[i].exx << " " << orientation[i].exy << " " << orientation[i].exz << std::endl;
        infile << orientation[i].eyx << " " << orientation[i].eyy << " " << orientation[i].eyz << std::endl;
        infile << orientation[i].ezx << " " << orientation[i].ezy << " " << orientation[i].ezz << std::endl;
        #endif
    }
    delete [] I;
}
