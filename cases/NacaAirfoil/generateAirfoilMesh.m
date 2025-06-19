clear;

% Geometry generation

x = [0:0.001:0.01 0.02:0.01:1.0];
t = 0.12;
yTop = 5 * t * (0.2969 * sqrt(x) - 0.1260 * x - 0.3516 * x.^2 + 0.2843 * x.^3 - 0.1036 * x.^4);
yBottom = -yTop;
plot(x, yTop, 'b-', x, yBottom, 'b-'),axis equal

[fid msg] = fopen("airfoilScript.py", 'w');

fprintf(fid, '#!/usr/bin/env python\n\n');

fprintf(fid, 'import sys\n');
fprintf(fid, 'import salome\n\n');

fprintf(fid, 'salome.salome_init()\n');
fprintf(fid, 'import salome_notebook\n');
fprintf(fid, 'notebook = salome_notebook.NoteBook()\n');

fprintf(fid, 'import GEOM\n');
fprintf(fid, 'from salome.geom import geomBuilder\n');
fprintf(fid, 'import math\n');
fprintf(fid, 'import SALOMEDS\n\n');

fprintf(fid, 'geompy = geomBuilder.New()\n\n');

fprintf(fid, 'O = geompy.MakeVertex(0, 0, 0)\n');
fprintf(fid, 'OX = geompy.MakeVectorDXDYDZ(1, 0, 0)\n');
fprintf(fid, 'OY = geompy.MakeVectorDXDYDZ(0, 1, 0)\n');
fprintf(fid, 'OZ = geompy.MakeVectorDXDYDZ(0, 0, 1)\n');

k = 0;
for vert = 1:length(x)
  ++k;
  fprintf(fid, 'v_%d = geompy.MakeVertex(%f, %f, 0)\n', k, x(vert), yTop(vert));
endfor

for vert = (length(x)-1):-1:2
  ++k;
  fprintf(fid, 'v_%d = geompy.MakeVertex(%f, %f, 0)\n', k, x(vert), yBottom(vert));
end

fprintf(fid, 'Polyline_1 = geompy.MakePolyline([');
for vert = 1:k
  fprintf(fid, 'v_%d, ', vert);
end

fprintf(fid, 'v_1])\n');

fprintf(fid, 'Face_1 = geompy.MakeFaceWires([Polyline_1], 1)\n');
fprintf(fid, 'Extrusion_1 = geompy.MakePrismVecH2Ways(Face_1, OZ, 1)\n');
fprintf(fid, 'v_A = geompy.MakeVertex(-3, -3, -3)\n');
fprintf(fid, 'v_B = geompy.MakeVertex(10, 3, 3)\n');
fprintf(fid, 'Box_1 = geompy.MakeBoxTwoPnt(v_A, v_B)\n');
fprintf(fid, 'Cut_1 = geompy.MakeCutList(Box_1, [Extrusion_1], True)\n');

fprintf(fid, 'geompy.addToStudy( O, "O" )\n');
fprintf(fid, 'geompy.addToStudy( OX, "OX" )\n');
fprintf(fid, 'geompy.addToStudy( OY, "OY" )\n');
fprintf(fid, 'geompy.addToStudy( OZ, "OZ" )\n');

for pt = 1:k
  fprintf(fid, 'geompy.addToStudy( v_%d, ''Vertex_%d'' )\n', pt, pt);
endfor

fprintf(fid, 'geompy.addToStudy(Polyline_1, "Polyline1")\n');
fprintf(fid, 'geompy.addToStudy( Face_1, "Face_1" )\n');
fprintf(fid, 'geompy.addToStudy( Extrusion_1, "Extrusion_1" )\n');
fprintf(fid, 'geompy.addToStudy( v_A, "v_A" )\n');
fprintf(fid, 'geompy.addToStudy( v_B, "v_B" )\n');
fprintf(fid, 'geompy.addToStudy( Box_1, "Box_1" )\n');
fprintf(fid, 'geompy.addToStudy( Cut_1, "Cut_1" )\n\n');

% Mesh generation

fprintf(fid, 'import  SMESH, SALOMEDS\n');
fprintf(fid, 'from salome.smesh import smeshBuilder\n\n');

fprintf(fid, 'smesh = smeshBuilder.New()\n\n');

fprintf(fid, 'Mesh_1 = smesh.Mesh(Cut_1,"Mesh_1")\n');
fprintf(fid, 'NETGEN_1D_2D_3D = Mesh_1.Tetrahedron(algo=smeshBuilder.NETGEN_1D2D3D)\n');
fprintf(fid, 'NETGEN_3D_Parameters_1 = NETGEN_1D_2D_3D.Parameters()\n');
fprintf(fid, 'NETGEN_3D_Parameters_1.SetSecondOrder( 0 )\n');
fprintf(fid, 'NETGEN_3D_Parameters_1.SetOptimize( 1 )\n');
fprintf(fid, 'NETGEN_3D_Parameters_1.SetFineness( 2 )\n');
fprintf(fid, 'NETGEN_3D_Parameters_1.SetChordalError( -1 )\n');
fprintf(fid, 'NETGEN_3D_Parameters_1.SetChordalErrorEnabled( 0 )\n');
fprintf(fid, 'NETGEN_3D_Parameters_1.SetUseSurfaceCurvature( 1 )\n');
fprintf(fid, 'NETGEN_3D_Parameters_1.SetFuseEdges( 1 )\n');
fprintf(fid, 'NETGEN_3D_Parameters_1.SetQuadAllowed( 0 )\n');
fprintf(fid, 'NETGEN_3D_Parameters_1.SetMinSize( 0.05 )\n');    % mesh parameter no. 1
fprintf(fid, 'NETGEN_3D_Parameters_1.SetMaxSize( 0.175 )\n');   % mesh parameter no. 2
fprintf(fid, 'NETGEN_3D_Parameters_1.SetCheckChartBoundary( 144 )\n\n');

fprintf(fid, 'smesh.SetName(NETGEN_3D_Parameters_1, "NETGEN 3D Parameters_1")\n');
fprintf(fid, 'smesh.SetName(Mesh_1.GetMesh(), "Mesh_1")\n');
fprintf(fid, 'smesh.SetName(NETGEN_1D_2D_3D.GetAlgorithm(), "NETGEN 1D-2D-3D")\n\n');

fprintf(fid, 'if salome.sg.hasDesktop():\n');
fprintf(fid, '  salome.sg.updateObjBrowser()\n');

fclose(fid);
