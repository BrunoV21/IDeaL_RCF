// Solve the Momentum equation

MRF.correctBoundaryVelocity(U);

tmp<fvVectorMatrix> tUEqn
(
    fvm::ddt(U) + fvm::div(phi, U)
  + MRF.DDt(U)
  //+ turbulence->divDevReff(U)
  - fvc::div((turbulence->nu()+nut_L)*dev2(T(fvc::grad(U))))
  - fvm::laplacian((turbulence->nu()+nut_L), U)
  + fvc::div(aperp)
 ==
    fvOptions(U)
);
fvVectorMatrix& UEqn = tUEqn.ref();

UEqn.relax();

fvOptions.constrain(UEqn);

if (pimple.momentumPredictor())
{
    solve(UEqn == -fvc::grad(p));

    fvOptions.correct(U);
}
