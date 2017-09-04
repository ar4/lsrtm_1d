import numpy as np
import scipy.optimize
from wave_1d_fd_pml.propagators import Pml2

class Lsrtm(object):
    def __init__(self, dx, dt=None, pml_width=10, profile=None):
        self.dx = dx
        self.dt = dt
        self.pml_width = pml_width
        self.profile = profile

    def born_model_shot(self, model, image, source_snapshots, receivers_x,
                        num_steps):
        """Born modeling/demigration."""
        assert image.ndim == 1
        assert source_snapshots.ndim == 2
        assert source_snapshots.shape[0] == num_steps
        assert source_snapshots.shape[1] == len(image)

        num_receivers = len(receivers_x)
        prop = Pml2(model, self.dx, self.dt, self.pml_width, self.profile)
        receivers = np.zeros([num_receivers, num_steps], np.float32)
        d2sdt2 = self._second_time_derivative(source_snapshots, self.dt)

        for step in range(1, num_steps-1):
            sources = (-image * d2sdt2[step, :])[:, np.newaxis]
            sources_x = np.arange(0, len(image))
            wavefield = prop.step(1, sources, sources_x)
            receivers[:, step] = wavefield[receivers_x]

        return receivers


    def migrate_shot(self, model, source, source_x, receivers, receivers_x,
                     maxiter=1, check_grad=False, manual_check_grad=False):
        """LSRTM main function"""
        assert source.ndim == 1
        assert receivers.ndim == 2
        source = source[np.newaxis, :]
        source_x = np.array([source_x])
        num_modeling_steps = receivers.shape[1]
        num_imaging_steps = num_modeling_steps - 1

        prop = Pml2(model, self.dx, self.dt, self.pml_width, self.profile)
        nx = len(model)

        source_snapshots = self._forward_source(source, source_x,
                                                1,
                                                num_modeling_steps, prop, nx)
        d2sdt2 = self._second_time_derivative(source_snapshots, self.dt)

        prop = Pml2(model, self.dx, self.dt, self.pml_width, self.profile)
        current_born_data = np.zeros([receivers.shape[0], receivers.shape[1]],
                                     np.float32)
        current_born_x = np.zeros(nx, np.float32)

        def born_data(x, current_born_x, current_born_data):
            """Call Born modeling if necessary."""
            if np.sum(np.abs(x - current_born_x)) == 0:
                return current_born_data
            else:
                current_born_x[:] = x
                current_born_data[:] = \
                        self.born_model_shot(model, x,
                                             source_snapshots, receivers_x,
                                             num_modeling_steps)
                return current_born_data

        def cost(x, cbx, cbd):
            """Calculate cost function."""
            return np.float32(0.5 * np.linalg.norm(born_data(x, cbx, cbd)
                                                   - receivers, 2)**2)
        def jac(x, cbx, cbd):
            """Calculate gradient/Jacobian."""
            return np.float32(self._backward_receivers( \
                                           (born_data(x, cbx, cbd) - receivers),
                                           receivers_x, 1,
                                           num_imaging_steps,
                                           d2sdt2, prop, nx))

        if manual_check_grad:
            costa = np.zeros(nx, np.float32)
            jaca = np.zeros(nx, np.float32)
            for change_x in range(nx):
                image = np.zeros_like(model)
                change_amp = 1e-12
                cost0 = cost(image, current_born_x, current_born_data)
                jac0 = jac(image, current_born_x, current_born_data)
                image[change_x] = change_amp
                cost1 = cost(image, current_born_x, current_born_data)
                jaca[change_x] = jac0[change_x]
                costa[change_x] = (cost1-cost0)/change_amp
            return costa, jaca

        if check_grad:
            return scipy.optimize.check_grad(cost, jac,
                                             np.zeros(nx, np.float32),
                                             current_born_x, current_born_data)

        res = scipy.optimize.minimize(cost, np.zeros(nx, np.float32),
                                      args=(current_born_x, current_born_data),
                                      jac=jac,
                                      options={'maxiter': maxiter, 'disp': True},
                                      bounds=[(np.float32(-2e-6),
                                               np.float32(2e-6))]*nx,
                                      method='TNC')

        return res

    def _forward_source(self, source, source_x,
                        imaging_condition_interval,
                        num_imaging_steps, prop, nx):
        """Forward propagate source waveform and save wavefield
        snapshots."""
        source_snapshots = np.zeros([num_imaging_steps, nx], np.float32)
        for imaging_step in range(0, num_imaging_steps):
            start_time_step = imaging_step * imaging_condition_interval
            end_time_step = start_time_step + imaging_condition_interval
            if end_time_step < source.shape[1]:
                source_snapshots[imaging_step, :] = \
                        prop.step(imaging_condition_interval,
                                  source[:, start_time_step:end_time_step],
                                  source_x)
            elif start_time_step < source.shape[1]:
                remaining_source_steps = source.shape[1] - start_time_step
                steps_after_source = (imaging_condition_interval -
                                      remaining_source_steps)
                prop.step(remaining_source_steps,
                          source[:, start_time_step:],
                          source_x)
                source_snapshots[imaging_step, :] = \
                        prop.step(steps_after_source)
            else:
                source_snapshots[imaging_step, :] = \
                        prop.step(imaging_condition_interval)

        return source_snapshots

    def _backward_receivers(self, receivers, receivers_x,
                            imaging_condition_interval,
                            num_imaging_steps,
                            source_snapshots, prop, nx):
        """RTM backpropagation and imaging condition."""
        image = np.zeros([nx], np.float32)
        for imaging_step in range(num_imaging_steps - 1, -1, -1):
            start_time_step = (imaging_step + 2) * imaging_condition_interval - 1
            end_time_step = start_time_step - imaging_condition_interval
            if start_time_step >= receivers.shape[1]:
                start_time_step = receivers.shape[1] - 1
            receiver_snapshot = \
                    prop.step(start_time_step - end_time_step,
                              -receivers[:, start_time_step:end_time_step:-1],
                              receivers_x)
            image += (source_snapshots[imaging_step, :] *
                      receiver_snapshot[:] * imaging_condition_interval)

        return image

    def _second_time_derivative(self, f, dt):
        """Calculate second time derivative of source snapshots."""
        d2fdt2 = np.zeros_like(f)
        d2fdt2[1:-1, :] = (f[2:, :] - 2*f[1:-1, :] + f[:-2, :]) / dt**2
        return d2fdt2

    def adjoint_test(self, model, source, nsteps):
        """Apply Claerbout's adjoint test. Both print statements
        should ideally print the same value,"""
        assert source.ndim == 1
        source = source[np.newaxis, :]
        source_x = np.array([1])
        nx = len(model)
        num_modeling_steps = nsteps
        num_imaging_steps = num_modeling_steps - 1
        prop = Pml2(model, self.dx, self.dt, self.pml_width, self.profile)
        source_snapshots = self._forward_source(source, source_x,
                                                1,
                                                num_modeling_steps, prop, nx)
        d2sdt2 = self._second_time_derivative(source_snapshots, self.dt)
        receivers_x = np.array([1])

        image0 = np.random.rand(nx).astype(np.float32)
        data0 = np.random.rand(1, nsteps).astype(np.float32)

        image1 = self._backward_receivers(data0, receivers_x, 1,
                                          num_imaging_steps,
                                          d2sdt2, prop, nx)
        data1 = self.born_model_shot(model, image0,
                                     source_snapshots, receivers_x,
                                     num_modeling_steps)

        print(image0 @ image1)
        print(data0[0, :] @ data1[0, :])


    def model_shot(self, model, source, source_x, receivers_x, max_time):
        """Generate synthetic data using true/exact model."""
        assert source.ndim == 1
        source = source[np.newaxis, :]
        source_x = np.array([source_x])
        num_receivers = len(receivers_x)

        prop = Pml2(model, self.dx, self.dt, self.pml_width, self.profile)

        nt = int(max_time / self.dt)
        receivers = np.zeros([num_receivers, nt], np.float32)
        for step in range(nt):
            wavefield = prop.step(1,
                                  source[:, step:step+1],
                                  source_x)
            receivers[:, step] = wavefield[receivers_x]

        return receivers
