#include <iostream>
#include <cmath>
#include <cstdlib>
#include <vector>

#include <armadillo>

using namespace std;
using namespace arma;

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

class Node
{
	public:
		double val, newval, integratedval;
};

class Link
{
	public:
		int src, dest;
		double weight;
};

class SubNet
{
	public:
		vector<Node> Nodes;
		vector<Link> Links;	
		
		mat LinkMatrix;
		vec upProjectionVector, downProjectionVector;
		
		void getWeightsFromInputs(vec Inputs);
		void makeRandomLinkMatrix(int inSize);
		void makeRandomProjection();
		
		void Iterate();
		double getProjection();
		double getInstantProjection();
		
		void generateFullyConnectedSubnet(int N, int inSize);
};

void SubNet::generateFullyConnectedSubnet(int N, int inSize)
{
	int i,j;
	
	for (i=0;i<N;i++)
	{
		Node thisNode;
		
		thisNode.val=(i==0);
		
		Nodes.push_back(thisNode);
	}
	
	for (i=0;i<N;i++)
		for (j=0;j<N;j++)
		{
			Link L;
			
			L.src = i;
			L.dest = j;
			L.weight = 0;
			
			Links.push_back(L);
		}

	makeRandomLinkMatrix(inSize); // How to compute this?
	makeRandomProjection();
}

double SubNet::getProjection()
{
	vec stateVec(Nodes.size());
	int i;
	
	for (i=0;i<Nodes.size();i++)
		stateVec.at(i) = Nodes[i].integratedval/10.0;
		
	return dot( upProjectionVector, stateVec );
}

double SubNet::getInstantProjection()
{
	vec stateVec(Nodes.size());
	int i;
	
	for (i=0;i<Nodes.size();i++)
		stateVec.at(i) = Nodes[i].val;
		
	return dot( downProjectionVector, stateVec );
}

void SubNet::Iterate()
{
	int i,j;
	
	for (i=0;i<Nodes.size();i++)
	{
		Nodes[i].integratedval += Nodes[i].val;
		Nodes[i].newval = 0;		
	}

	for (j=0;j<Links.size();j++)
	{
		Nodes[Links[j].dest].newval += Nodes[Links[j].src].val * Links[j].weight;
	}
	
	for (j=0;j<Nodes.size();j++)
	{
		Nodes[j].val = Nodes[j].newval;
	}
}

class Layer
{
	public:
		Layer *Lower, *Upper;
		vector<SubNet> S;
		vec Inputs;
		
		void Iterate();
		void generateLayerArchitecture(int N);
		void createInputVectors();
};

void Layer::generateLayerArchitecture(int N)
{
	int i;
	
	for (i=0;i<N;i++)
	{
		SubNet thisSubnet;
		
		S.push_back(thisSubnet);
	}
}

void Layer::createInputVectors()
{
	int len=1;
	
	if (Lower)
	{
		len += Lower->S.size();
	}
	
	if (Upper)
	{
		len += Upper->S.size();
	}
	
	Inputs = zeros<vec>(len);
}

void Layer::Iterate()
{
	int i, offset=0;
	
	if (Lower)
	{
		for (i=0;i<10;i++)
		{
			Lower->Iterate();
		}
				
		for (i=0;i<Lower->S.size();i++)
		{
			Inputs.at(i) = Lower->S[i].getProjection();
		}
		
		offset += Lower->S.size();
	}	
	
	if (Upper)
	{
		for (i=0;i<Upper->S.size();i++)
			Inputs.at(i+offset) = Upper->S[i].getInstantProjection();		
			
		offset += Upper->S.size();
	}
	
	Inputs.at(offset) = 1;
	
	for (i=0;i<S.size();i++)
	{
		S[i].getWeightsFromInputs(Inputs);
		S[i].Iterate();
	}
}

void SubNet::getWeightsFromInputs(vec Inputs)
{
	int i,j;
	vec Weights = LinkMatrix * Inputs;
	
	for (i=0;i<Links.size();i++)
	{
		Links[i].weight = fabs(Weights.at(i));
	}
	
	for (i=0;i<Nodes.size();i++)
	{
		double total=1e-20;
		
		for (j=0;j<Links.size();j++)
			if (Links[j].src == i)
				total += Links[j].weight;
				
		for (j=0;j<Links.size();j++)
			if (Links[j].src == i)
				Links[j].weight /= total;
	}
}

void SubNet::makeRandomLinkMatrix(int inSize)
{
	int i,j;
	
	LinkMatrix = zeros<mat>(Links.size(), inSize); 
	
	for (i = 0; i < inSize ; i++)
	{
		for (j = 0; j < Links.size(); j++)
		{
			LinkMatrix.at(j,i) = (rand()%2000001-1000000.0)/1000000.0;
		}
	}
}

void SubNet::makeRandomProjection()
{
	int i;
	
	upProjectionVector = zeros<vec>(Nodes.size());
	downProjectionVector = zeros<vec>(Nodes.size());
	
	for (i=0;i<Nodes.size();i++)
	{
		upProjectionVector.at(i) = (rand()%2000001-1000000.0)/1000000.0;		
		downProjectionVector.at(i) = (rand()%2000001-1000000.0)/1000000.0;		
	}
}

Layer Root;

void InitNetwork()
{
	Root.Lower = new Layer();
	Root.Lower->Lower = NULL;
	Root.Lower->Upper = &Root;
	Root.Upper = NULL;
	
	Root.generateLayerArchitecture(1);
	Root.Lower->generateLayerArchitecture(2);
	
	Root.createInputVectors();
	Root.Lower->createInputVectors();
	
	int i;
	
	for (i=0;i<Root.S.size();i++)
	{
		Root.S[i].generateFullyConnectedSubnet(2, Root.Inputs.size());
	}
	
	for (i=0;i<Root.Lower->S.size();i++)
	{
		Root.Lower->S[i].generateFullyConnectedSubnet(2, Root.Lower->Inputs.size());
	}
}

int main(int argc, char **argv)
{
	InitNetwork();
	
	int i;
	
	for (i=0;i<100;i++)
	{
		Root.Iterate();
		
		FILE *f=fopen("timeseries.txt","a");
		int j,k;
		
		fprintf(f,"%d ",i);
		for (j=0;j<Root.S.size();j++)
		{
			for (k=0;k<Root.S[j].Nodes.size();k++)
			{
				fprintf(f,"%.6g ",Root.S[j].Nodes[k].val);
			}
		}

		for (j=0;j<Root.Lower->S.size();j++)
		{
			for (k=0;k<Root.Lower->S[j].Nodes.size();k++)
			{
				fprintf(f,"%.6g ",Root.Lower->S[j].Nodes[k].val);
			}
		}
		
		fprintf(f,"\n");
		fclose(f);		
	}
}
