"""
OpenMM Reporter Adapter Module
"""

#####Imports#####
from coddiwomple.reporters import Reporter
import mdtraj.utils as mdtrajutils
import mdtraj as md
import os
import numpy as np
import logging

#####Instantiate Logger#####
logging.basicConfig(level = logging.NOTSET)
_logger = logging.getLogger("openmm_reporters")
_logger.setLevel(logging.DEBUG)

class OpenMMReporter(Reporter):
    """
    OpenMM specific reporter
    """
    def __init__(self,
                 trajectory_directory,
                 trajectory_prefix,
                 topology,
                 subset_indices = None,
                 **kwargs):
        """
        Initialize the OpenMM particles.

        arguments
            trajectory_directory : str
                name of directory
            trajectory_prefix : str
                prefix of the files to write
            topology : simtk.openmm.app.topology.Topology
                (complete) topology object to which to write
            subset_indices : list(int)
                zero-indexed atom indices to write
        """
        # equip attributes
        self.trajectory_directory, self.trajectory_prefix = trajectory_directory, trajectory_prefix
        self.topology = topology
        self.hex_dict = {}

        #prepare topology object
        if self.trajectory_directory is not None and self.trajectory_prefix is not None:
            _logger.debug(f"creating trajectory storage object...")
            self.write_traj = True
            self.neq_traj_filename = os.path.join(os.getcwd(), self.trajectory_directory,
                                                  f"{self.trajectory_prefix}.neq")
            os.mkdir(os.path.join(os.getcwd(), self.trajectory_directory))
            md_topology = md.Topology().from_openmm(self.topology)
            if subset_indices is None:
                self.md_topology = md_topology
                self.subset_indices = range(self.topology.getNumAtoms())
            else:
                self.md_topology = complex_md_topology.md_topology.subset(subset_indices)
                self.subset_indices = subset_indices
        else:
            self.write_traj = False

    def record(self, particles, save_to_disk = False, **kwargs):
        """
        append the positions, box lengths, and box angles to their respective attributes and save to disk if specified;
        save to disk if specified

        arguments
            particles : list(coddiwomple.particles.Particle)
                list of particle objects
            save_to_disk : bool, default False
                whether to save the trajectory to disk
        """
        if self.write_traj:
            for particle in particles:
                particle_hex = hex(id(particle))
                if particle_hex in self.hex_dict.keys():
                    pass
                else:
                    self.hex_dict[particle_hex] = [[], [], []]

                self.hex_dict[particle_hex][0].append(particle.state.positions[self.subset_indices, :].value_in_unit_system(unit.md_unit_system))
                a, b, c, alpha, beta, gamma = mdtrajutils.unitcell.box_vectors_to_lengths_and_angles(*particle.state.box_vectors)
                self.hex_dict[particle_hex][1].append([a, b, c])
                self.hex_dict[particle_hex][2].append([alpha, beta, gamma])

                if save_to_disk:
                    filename = f"{self.neq_traj_filename}.{particle_hex}.pdb"
                    self._write_trajectory(hex = particle_hex, filename = filename)

    def _write_trajectory(self, hex, filename):
        """
        write a trajectory given a filename.

        arguments
            filename : str
                name of the file to write
            hex : str
                hex memory address
        """
        traj = md.Trajectory(xyz = np.array(self.hex_dict[hex][0]),
                             unitcell_lengths = np.array(self.hex_dict[hex][1]),
                             unitcell_angles = np.array(self.hex_dict[hex][2]),
                             topology = self.md_topology,
                             )
        traj.center_coordinates()
        #traj.image_molecules(inplace=True)
        traj.save(filename)
