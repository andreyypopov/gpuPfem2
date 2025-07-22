#include "data_export_3d.cuh"

DataExport3D::DataExport3D(const Mesh3D &mesh, const ParticleHandler3D *particleHandler)
    : mesh(mesh)
    , particleHandler(particleHandler)
    , particleCount(0)
{

}

void DataExport3D::addScalarDataVector(const deviceVector<double> &dataVector, const std::string &fieldname)
{
    scalarDataVectors[fieldname] = dataVector.data;
    hostScalarDataVectors[fieldname].resize(mesh.getHostVertices().size());
}

void DataExport3D::addVectorDataVector(const deviceVector<double*> &dataVector, const std::string &fieldname)
{
    std::vector<double*> hostPointers(3);
    copy_d2h(dataVector.data, hostPointers.data(), 3);

    for (int i = 0; i < 3; ++i){
        vectorDataVectors[fieldname][i] = hostPointers[i];
        hostVectorDataVectors[fieldname][i].resize(mesh.getHostVertices().size());
    }
}

void DataExport3D::addTensorDataVector(const deviceVector<GenericMatrix3x3> &dataVector, const std::string &fieldname)
{
    tensorDataVectors[fieldname] = dataVector.data;
    hostTensorDataVectors[fieldname].resize(mesh.getHostVertices().size());
}

void DataExport3D::exportToVTK(const std::string &filename) const
{
    std::ofstream outputFile(filename.c_str());
    if(outputFile.is_open()){
        const auto &hostVertices = mesh.getHostVertices();
        const auto &hostCells = mesh.getHostCells();

        //header
        outputFile << "<?xml version=\"1.0\" ?> " << std::endl;
        outputFile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
        outputFile << "  <UnstructuredGrid>" << std::endl;
        outputFile << "    <Piece NumberOfPoints=\"" << hostVertices.size() <<  "\" NumberOfCells=\"" << hostCells.size() << "\">" << std::endl;

        //vertices
        outputFile << "      <Points>" << std::endl;
        outputFile << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" Format=\"ascii\">" << std::endl;
        for(const Point3 &vertex : hostVertices)
            outputFile << "        " << vertex.x << " " << vertex.y << " " << vertex.z << std::endl;
        outputFile << "        </DataArray>" << std::endl;
        outputFile << "      </Points>" << std::endl;

        //cells (tetrahedra)
        outputFile << "      <Cells>" << std::endl;
        outputFile << "        <DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">" << std::endl;
        outputFile << "          ";
        for (const uint4 &cell : hostCells)
            outputFile << cell.x << " " << cell.y << " " << cell.z << " " << cell.w << " ";
        outputFile << std::endl;
        outputFile << "        </DataArray>" << std::endl;

        //offsets
        outputFile << "        <DataArray type=\"Int32\" Name=\"offsets\" Format=\"ascii\">" << std::endl;
        outputFile << "          ";
        for (int i = 0; i < hostCells.size(); ++i)
            outputFile << (i + 1) * 4 << " ";
        outputFile << std::endl;
        outputFile << "        </DataArray>" << std::endl;

        //cell types
        outputFile << "        <DataArray type=\"UInt8\" Name=\"types\" Format=\"ascii\">" << std::endl;
        outputFile << "          ";
        for (int i = 0; i < hostCells.size(); ++i)
            outputFile << 10 << " ";
        outputFile << std::endl;
        outputFile << "        </DataArray>" << std::endl;

        outputFile << "      </Cells>" << std::endl;

        const bool fieldsAreUsed = !scalarDataVectors.empty() || !vectorDataVectors.empty() || !tensorDataVectors.empty();
        if (fieldsAreUsed)
            outputFile << "      <PointData Scalars=\"scalars\">" << std::endl;

        if(!scalarDataVectors.empty())
            for(const auto& it : scalarDataVectors){
                outputFile << "        <DataArray type=\"Float32\" Name=\"" << it.first << "\" Format=\"ascii\">" << std::endl;
                outputFile << "        ";

                const double *hostData = hostScalarDataVectors.at(it.first).data();

                copy_d2h(it.second, hostData, hostVertices.size());

                for(int i = 0; i < hostVertices.size(); ++i)
                    outputFile << hostData[i] << "  ";
                outputFile << std::endl;

                outputFile << "        </DataArray>" << std::endl;
            }

        if (!vectorDataVectors.empty())
            for(const auto& it : vectorDataVectors) {
                outputFile << "        <DataArray type=\"Float32\" Name=\"" << it.first << "\" NumberOfComponents=\"3\" Format=\"ascii\">" << std::endl;

                const auto &hostData = hostVectorDataVectors.at(it.first);
                for (int i = 0; i < 3; ++i)
                    copy_d2h(it.second[i], hostData[i].data(), hostVertices.size());

                for(int i = 0; i < hostVertices.size(); ++i)
                    outputFile << "          " << hostData[0][i] << " " << hostData[1][i] << " " << hostData[2][i] << std::endl;

                outputFile << "        </DataArray>" << std::endl;
            }

        if (!tensorDataVectors.empty())
            for(const auto& it : tensorDataVectors) {
                outputFile << "        <DataArray type=\"Float32\" Name=\"" << it.first << "\" NumberOfComponents=\"9\" Format=\"ascii\">" << std::endl;

                const GenericMatrix3x3 *hostData = hostTensorDataVectors.at(it.first).data();
                copy_d2h(it.second, hostData, hostVertices.size());

                for(int i = 0; i < hostVertices.size(); ++i){
                    outputFile << "          ";
                    for(int k = 0; k < 3; ++k)
                        for(int l = 0; l < 3; ++l)
                            outputFile << hostData[i](k, l) << " ";
                    outputFile << std::endl;
                }

                outputFile << "        </DataArray>" << std::endl;
            }

        if (fieldsAreUsed)
            outputFile << "      </PointData>" << std::endl;

        //footer
        outputFile << "    </Piece>" << std::endl;
        outputFile << "  </UnstructuredGrid>" << std::endl;
        outputFile << "</VTKFile>" << std::endl;

        outputFile.close();
        printf("Mesh solution saved to %s\n", filename.c_str());
    } else
        printf("Error while saving mesh solution to a file\n");
}

void DataExport3D::exportParticlesToVTK(const std::string & filename)
{
    if(particleCount != particleHandler->getParticleCount()){
        particleCount = particleHandler->getParticleCount();
        hostParticles.resize(particleCount);
    }

    copy_d2h(particleHandler->getParticles(), hostParticles.data(), particleCount);

    std::ofstream outputFile(filename.c_str());
    if(outputFile.is_open()){
        //header
        outputFile << "<?xml version=\"1.0\" ?> " << std::endl;
        outputFile << "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"LittleEndian\">" << std::endl;
        outputFile << "  <UnstructuredGrid>" << std::endl;
        outputFile << "    <Piece NumberOfPoints=\"" << particleCount <<  "\" NumberOfCells=\"" << particleCount << "\">" << std::endl;

        //positions
    	outputFile << "      <Points>" << std::endl;
	    outputFile << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" Format=\"ascii\">" << std::endl;
	    for(const auto &particleIndex : hostParticles)
		    outputFile << "          " << particleIndex.getPosition().x << " " << particleIndex.getPosition().y << " " << particleIndex.getPosition().z << std::endl;

	    outputFile << "        </DataArray>" << std::endl;
    	outputFile << "      </Points>" << std::endl;

        //cells (equal to particles)
        outputFile << "      <Cells>" << std::endl;
        outputFile << "        <DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">" << std::endl;
        outputFile << "        ";
        for (int i = 0; i < particleCount; ++i)
            outputFile << "  " << i;
        
        outputFile << std::endl;
        outputFile << "        </DataArray>" << std::endl;

        //offsets
        outputFile << "        <DataArray type=\"Int32\" Name=\"offsets\" Format=\"ascii\">" << std::endl;
        outputFile << "        ";
        for (int i = 0; i < particleCount; ++i)
            outputFile << "  " << i + 1; 
        
        outputFile << std::endl;
        outputFile << "        </DataArray>" << std::endl;

        outputFile << "        <DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">" << std::endl;
        outputFile << "        ";
        for (int i = 0; i < particleCount; ++i)
            outputFile << "  " << 1;
        
        outputFile << std::endl;
        outputFile << "        </DataArray>" << std::endl;
        outputFile << "      </Cells>" << std::endl;

        //data in particles (field values)
	    outputFile << "      <PointData Scalars=\"scalars\">" << std::endl;
	
        //velocity
        outputFile << "        <DataArray type=\"Float32\" Name=\"velocity\" NumberOfComponents=\"3\" Format=\"ascii\">" << std::endl;
        for(const auto &particleIndex : hostParticles)
            outputFile << "          " << particleIndex.getVelocity().x << " " << particleIndex.getVelocity().y << " " << particleIndex.getVelocity().z << std::endl;
        outputFile << "        </DataArray>" << std::endl;

        outputFile << "      </PointData>" << std::endl;

        //footer
        outputFile << "    </Piece>" << std::endl;
        outputFile << "  </UnstructuredGrid>" << std::endl;
        outputFile << "</VTKFile>" << std::endl;

        outputFile.close();
        printf("Particles saved to %s\n", filename.c_str());
    } else
        printf("Error while saving particles to a file\n");
}
