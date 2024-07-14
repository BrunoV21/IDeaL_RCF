Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.timeName(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field aperp\n" << endl ;
const volSymmTensorField aperp
(
    IOobject
    (
	"aperp" ,
	runTime.timeName(),
	mesh,
	IOobject::MUST_READ ,
	IOobject::NO_WRITE
    ),
    mesh
);

Info<< "Reading field nut_L\n" << endl ;
const volScalarField nut_L
(
    IOobject
    (
	"nut_L" ,
	runTime.timeName(),
	mesh,
	IOobject::MUST_READ ,
	IOobject::NO_WRITE
    ),
    mesh
);

#include "createPhi.H"


label pRefCell = 0;
scalar pRefValue = 0.0;
setRefCell(p, simple.dict(), pRefCell, pRefValue);
mesh.setFluxRequired(p.name());


singlePhaseTransportModel laminarTransport(U, phi);

autoPtr<incompressible::turbulenceModel> turbulence
(
    incompressible::turbulenceModel::New(U, phi, laminarTransport)
);

#include "createMRF.H"
#include "createFvOptions.H"
