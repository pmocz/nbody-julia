"""
Create Your Own N-body Simulation (With Julia)
Philip Mocz (2020) Princeton Univeristy, @PMocz

Simulate orbits of stars interacting due to gravity
Code calculates pairwise forces according to Newton's Law of Gravity
"""

using Random
using Statistics
using Plots
using LinearAlgebra

"""
Calculate the acceleration on each particle due to Newton's Law
pos  is an N x 3 matrix of positions
mass is an N x 1 vector of masses
G is Newton's Gravitational constant
softening is the softening length
a is N x 3 matrix of accelerations
"""
function getAcc( pos, mass, G, softening )
	# positions r = [x,y,z] for all particles
	x = pos[:,1];
	y = pos[:,2];
	z = pos[:,3];

	# matrix that stores all pairwise particle separations: r_j - r_i
	dx = x' .- x;
	dy = y' .- y;
	dz = z' .- z;

	# matrix that stores 1/r^3 for all particle pairwise particle separations
	inv_r3 = (dx.^2 + dy.^2 + dz.^2 .+ softening.^2);
	inv_r3[inv_r3.>0] = inv_r3[inv_r3.>0].^(-1.5);

	ax = G .* (dx .* inv_r3) * mass;
	ay = G .* (dy .* inv_r3) * mass;
	az = G .* (dz .* inv_r3) * mass;

	# pack together the acceleration components
	a = [ax ay az];

	return a;
end

"""
Get kinetic energy (KE) and potential energy (PE) of simulation
pos is N x 3 matrix of positions
vel is N x 3 matrix of velocities
mass is an N x 1 vector of masses
G is Newton's Gravitational constant
KE is the kinetic energy of the system
PE is the potential energy of the system
"""
function getEnergy( pos, vel, mass, G )
	# Kinetic Energy:
	KE = 0.5*sum(mass .* vel.^2);

	# Potential Energy:

	# positions r = [x,y,z] for all particles
	x = pos[:,1];
	y = pos[:,2];
	z = pos[:,3];

	# matrix that stores all pairwise particle separations: r_j - r_i
	dx = x' .- x;
	dy = y' .- y;
	dz = z' .- z;

	# matrix that stores r for all particle pairwise particle separations
	r = (dx.^2 + dy.^2 + dz.^2).^(0.5);

	# sum over upper triangle, to count each interaction only once
	PE = G * sum(sum(triu(-(mass*mass')./r,1)));

	return KE, PE;
end


""" N-body simulation """
function main()
	# Simulation parameters
	N         = 100;    # Number of particles
	t         = 0;      # current time of the simulation
	tEnd      = 10;     # time at which simulation ends
	dt        = 0.01;   # timestep
	softening = 0.1;    # softening length
	G         = 1;      # Newton's Gravitational Constant
	plotRealTime = false;  # switch on for plotting as the simulation goes along

	# Generate Initial Conditions
	rng = MersenneTwister(42);    # set the random number generator seed

	mass = 20*ones(N,1)/N;        # total mass of particles is 20
	pos  = randn(rng,N,3);        # randomly selected positions and velocities
	vel  = randn(rng,N,3);

	# Convert to Center-of-Mass frame
	vel .-= mean(mass .* vel, dims=1) ./ mean(mass);

	# calculate initial gravitational accelerations
	acc = getAcc( pos, mass, G, softening );
	
	# calculate initial energy of system
	KE, PE  = getEnergy( pos, vel, mass, G );

	# number of timesteps
	Nt = convert(Int64,ceil(tEnd/dt));

	# save particle orbits for plotting trails
	pos_save = zeros(N,3,Nt+1);
	pos_save[:,:,1] = pos;
	KE_save = zeros(Nt+1,1);
	KE_save[1] = KE;
	PE_save = zeros(Nt+1,1);
	PE_save[1] = PE;
	t_all = [i for  i=0:Nt] * dt;

	# prep plot
	gr(size=(400,500), legend=false, dpi = 240, markerstrokewidth=0.01, markerstrokecolor=:white)
	fig = plot(1)


	# Simulation Main Loop
	for i in 1:Nt
		# (1/2) kick
		vel += acc * dt/2;

		# drift
		pos += vel * dt;

		# update accelerations
		acc = getAcc( pos, mass, G, softening );

		# (1/2) kick
		vel += acc * dt/2;

		# update time
		t += dt;

		# get energy of system
		KE, PE  = getEnergy( pos, vel, mass, G );
		KE_save[i+1] = KE;
		PE_save[i+1] = PE;

		# save energies, positions for plotting trail
		pos_save[:,:,i+1] = pos

		# plot in real time
		if (plotRealTime) || (i==Nt)
			xx = pos_save[:,1,max(i-50,1):i]
			yy = pos_save[:,2,max(i-50,1):i]
			plt1 = plot(xx, yy, seriestype=:scatter, markersize=1, color=RGB(.7,.7,1), xlim = (-2,2),ylim=(-2,2), aspect_ratio = :equal)
			plt1 = plot!(plt1, pos[:,1], pos[:,2], seriestype=:scatter, markersize=4, color=:blue)
			plt2 = plot([], seriestype=:scatter,xlim = (-0,tEnd),ylim=(-300,300), xlabel="time", ylabel="energy",label="", legend=true)
			plt2 = plot!(plt2, t_all, KE_save, seriestype=:scatter, markersize=1, color=:red, label=(i==Nt) ? "KE" : "" )
			plt2 = plot!(plt2, t_all, PE_save, seriestype=:scatter, markersize=1, color=:blue, label=(i==Nt) ? "PE" : "" )
			plt2 = plot!(plt2, t_all, KE_save+PE_save, seriestype=:scatter, markersize=1, color=:black, label=(i==Nt) ? "Etot" : "" )
			fig = plot(plt1,plt2, layout=grid(2,1, heights=[.8,.2]))
			display(fig)
		end

	end

	# Save Figure
	savefig(fig,"nbody.png")

end

main()
