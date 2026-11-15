//! The FLIP solver: particles carry the water, a staggered (MAC) grid makes
//! it incompressible. Each step:
//!
//! 1. mark the cells that hold particles as water;
//! 2. move particle velocities onto the grid faces (trilinear weights);
//! 3. add the field's acceleration, face by face;
//! 4. solve for pressure (preconditioned conjugate gradients) so no water
//!    cell gains or loses volume, and subtract its gradient;
//! 5. give each particle the grid's change in velocity (FLIP), blended with
//!    a little of the grid velocity itself (PIC) for calm;
//! 6. move particles through the grid velocity, keep them out of rock.
//!
//! Nothing here assumes a direction: the acceleration can point anywhere,
//! differ from cell to cell, or be zero, so water pools on walls, hangs
//! under ceilings and floats as blobs where nothing pulls.

use glam::{IVec3, UVec3, Vec3};

/// What fills a cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Cell {
    Air,
    Water,
    Solid,
}

/// The solver's state. Positions are in the fluid's own space, with the
/// box's minimum corner at the origin (`World` and [`super::Fluid`] add the
/// box origin).
#[derive(Clone, Debug)]
pub(crate) struct Sim {
    pub n: UVec3,
    pub h: f32,
    pub solid: Vec<bool>,
    pub cells: Vec<Cell>,
    pub u: Vec<f32>,
    pub v: Vec<f32>,
    pub w: Vec<f32>,
    u_old: Vec<f32>,
    v_old: Vec<f32>,
    w_old: Vec<f32>,
    pub pos: Vec<Vec3>,
    pub vel: Vec<Vec3>,
    pressure: Vec<f32>,
    /// Acceleration at each cell centre, refreshed by the caller.
    pub accel: Vec<Vec3>,
    pub flip: f32,
    /// Velocity damping per second inside the water (bed and air drag).
    pub drag: f32,
    /// Particles that left the box since the last call to `take_lost`.
    pub lost: usize,
    pub last_iterations: usize,
    /// Step length of the last pressure solve (for warm starts).
    last_dt: f32,
}

fn idx(n: UVec3, x: u32, y: u32, z: u32) -> usize {
    (x + n.x * (y + n.y * z)) as usize
}

impl Sim {
    pub fn new(n: UVec3, h: f32) -> Self {
        let cells = (n.x * n.y * n.z) as usize;
        let fu = ((n.x + 1) * n.y * n.z) as usize;
        let fv = (n.x * (n.y + 1) * n.z) as usize;
        let fw = (n.x * n.y * (n.z + 1)) as usize;
        Self {
            n,
            h,
            solid: vec![false; cells],
            cells: vec![Cell::Air; cells],
            u: vec![0.0; fu],
            v: vec![0.0; fv],
            w: vec![0.0; fw],
            u_old: vec![0.0; fu],
            v_old: vec![0.0; fv],
            w_old: vec![0.0; fw],
            pos: Vec::new(),
            vel: Vec::new(),
            pressure: vec![0.0; cells],
            accel: vec![Vec3::ZERO; cells],
            flip: 0.95,
            drag: 0.05,
            lost: 0,
            last_iterations: 0,
            last_dt: 0.0,
        }
    }

    pub fn cell_index(&self, c: IVec3) -> Option<usize> {
        if c.cmplt(IVec3::ZERO).any() || c.cmpge(self.n.as_ivec3()).any() {
            return None;
        }
        Some(idx(self.n, c.x as u32, c.y as u32, c.z as u32))
    }

    pub fn cell_of(&self, p: Vec3) -> IVec3 {
        (p / self.h).floor().as_ivec3()
    }

    /// Solid outside the box, and where marked inside.
    pub fn is_solid(&self, c: IVec3) -> bool {
        self.cell_index(c).is_none_or(|i| self.solid[i])
    }

    pub fn cell_center(&self, c: IVec3) -> Vec3 {
        (c.as_vec3() + 0.5) * self.h
    }

    // --- faces ------------------------------------------------------------

    /// Face grid dimensions and the offset of its samples, per component.
    fn face_dims(&self, axis: usize) -> (UVec3, Vec3) {
        let n = self.n;
        match axis {
            0 => (UVec3::new(n.x + 1, n.y, n.z), Vec3::new(0.0, 0.5, 0.5)),
            1 => (UVec3::new(n.x, n.y + 1, n.z), Vec3::new(0.5, 0.0, 0.5)),
            _ => (UVec3::new(n.x, n.y, n.z + 1), Vec3::new(0.5, 0.5, 0.0)),
        }
    }

    fn faces(&self, axis: usize) -> &Vec<f32> {
        match axis {
            0 => &self.u,
            1 => &self.v,
            _ => &self.w,
        }
    }

    /// Trilinear sample of one face component at `p`.
    fn sample_axis(&self, data: &[f32], axis: usize, p: Vec3) -> f32 {
        let (dims, off) = self.face_dims(axis);
        let g = p / self.h - off;
        // Keep the 2x2x2 stencil inside the face grid.
        let hi = (dims.as_vec3() - 2.0).max(Vec3::ZERO);
        let i = g.floor().clamp(Vec3::ZERO, hi);
        let f = (g - i).clamp(Vec3::ZERO, Vec3::ONE);
        let i = i.as_uvec3();
        let sx = 1usize;
        let sy = dims.x as usize;
        let sz = (dims.x * dims.y) as usize;
        let base = i.x as usize + sy * i.y as usize + sz * i.z as usize;
        let (ox, oy, oz) = (
            if dims.x > 1 { sx } else { 0 },
            if dims.y > 1 { sy } else { 0 },
            if dims.z > 1 { sz } else { 0 },
        );
        let c00 = data[base] * (1.0 - f.x) + data[base + ox] * f.x;
        let c10 = data[base + oy] * (1.0 - f.x) + data[base + oy + ox] * f.x;
        let c01 = data[base + oz] * (1.0 - f.x) + data[base + oz + ox] * f.x;
        let c11 = data[base + oz + oy] * (1.0 - f.x) + data[base + oz + oy + ox] * f.x;
        let c0 = c00 * (1.0 - f.y) + c10 * f.y;
        let c1 = c01 * (1.0 - f.y) + c11 * f.y;
        c0 * (1.0 - f.z) + c1 * f.z
    }

    pub fn velocity_at(&self, p: Vec3) -> Vec3 {
        Vec3::new(
            self.sample_axis(&self.u, 0, p),
            self.sample_axis(&self.v, 1, p),
            self.sample_axis(&self.w, 2, p),
        )
    }

    fn old_velocity_at(&self, p: Vec3) -> Vec3 {
        Vec3::new(
            self.sample_axis(&self.u_old, 0, p),
            self.sample_axis(&self.v_old, 1, p),
            self.sample_axis(&self.w_old, 2, p),
        )
    }

    // --- steps ------------------------------------------------------------

    fn classify(&mut self) {
        for (c, s) in self.cells.iter_mut().zip(&self.solid) {
            *c = if *s { Cell::Solid } else { Cell::Air };
        }
        for k in 0..self.pos.len() {
            let c = self.cell_of(self.pos[k]);
            if let Some(i) = self.cell_index(c) {
                if self.cells[i] == Cell::Air {
                    self.cells[i] = Cell::Water;
                }
            }
        }
    }

    fn particles_to_grid(&mut self) {
        use rayon::prelude::*;
        let this = &*self;
        let mut out: Vec<Vec<f32>> = (0..3usize)
            .into_par_iter()
            .map(|axis| this.splat_axis(axis))
            .collect();
        self.w = out.pop().expect("3");
        self.v = out.pop().expect("3");
        self.u = out.pop().expect("3");
    }

    fn splat_axis(&self, axis: usize) -> Vec<f32> {
        let (dims, off) = self.face_dims(axis);
        let len = (dims.x * dims.y * dims.z) as usize;
        let mut sum = vec![0.0f32; len];
        let mut weight = vec![0.0f32; len];
        let hi = dims.as_ivec3() - 1;
        let sy = dims.x as usize;
        let sz = (dims.x * dims.y) as usize;
        for k in 0..self.pos.len() {
            let g = self.pos[k] / self.h - off;
            let i = g.floor().as_ivec3();
            let f = g - i.as_vec3();
            let value = self.vel[k][axis];
            for (dz, wz) in [(0, 1.0 - f.z), (1, f.z)] {
                let z = i.z + dz;
                if z < 0 || z > hi.z {
                    continue;
                }
                for (dy, wy) in [(0, 1.0 - f.y), (1, f.y)] {
                    let y = i.y + dy;
                    if y < 0 || y > hi.y {
                        continue;
                    }
                    for (dx, wx) in [(0, 1.0 - f.x), (1, f.x)] {
                        let x = i.x + dx;
                        if x < 0 || x > hi.x {
                            continue;
                        }
                        let j = x as usize + sy * y as usize + sz * z as usize;
                        let wgt = wx * wy * wz;
                        sum[j] += value * wgt;
                        weight[j] += wgt;
                    }
                }
            }
        }
        for j in 0..len {
            sum[j] = if weight[j] > 1e-6 {
                sum[j] / weight[j]
            } else {
                0.0
            };
        }
        sum
    }

    /// Whether the face between cell `a` and the next cell along `axis`
    /// touches water (and so takes part in the step).
    fn face_cells(&self, axis: usize, f: UVec3) -> (IVec3, IVec3) {
        let hi = f.as_ivec3();
        let mut lo = hi;
        lo[axis] -= 1;
        (lo, hi)
    }

    fn cell_kind(&self, c: IVec3) -> Cell {
        self.cell_index(c).map_or(Cell::Solid, |i| self.cells[i])
    }

    fn apply_forces(&mut self, dt: f32) {
        let damp = (-self.drag * dt).exp();
        for axis in 0..3 {
            let (dims, _) = self.face_dims(axis);
            for z in 0..dims.z {
                for y in 0..dims.y {
                    for x in 0..dims.x {
                        let f = UVec3::new(x, y, z);
                        let (a, b) = self.face_cells(axis, f);
                        let (ka, kb) = (self.cell_kind(a), self.cell_kind(b));
                        if ka != Cell::Water && kb != Cell::Water {
                            continue;
                        }
                        let acc = |c: IVec3| self.cell_index(c).map(|i| self.accel[i][axis]);
                        let g = match (acc(a), acc(b)) {
                            (Some(p), Some(q)) => 0.5 * (p + q),
                            (Some(p), None) | (None, Some(p)) => p,
                            (None, None) => 0.0,
                        };
                        let j = (x + dims.x * (y + dims.y * z)) as usize;
                        let faces = match axis {
                            0 => &mut self.u,
                            1 => &mut self.v,
                            _ => &mut self.w,
                        };
                        faces[j] = (faces[j] + g * dt) * damp;
                    }
                }
            }
        }
    }

    /// Faces against rock (or the box) carry no flow through them.
    fn enforce_solids(&mut self) {
        for axis in 0..3 {
            let (dims, _) = self.face_dims(axis);
            for z in 0..dims.z {
                for y in 0..dims.y {
                    for x in 0..dims.x {
                        let (a, b) = self.face_cells(axis, UVec3::new(x, y, z));
                        if self.is_solid(a) || self.is_solid(b) {
                            let j = (x + dims.x * (y + dims.y * z)) as usize;
                            match axis {
                                0 => self.u[j] = 0.0,
                                1 => self.v[j] = 0.0,
                                _ => self.w[j] = 0.0,
                            }
                        }
                    }
                }
            }
        }
    }

    fn face_value(&self, axis: usize, f: IVec3) -> f32 {
        let (dims, _) = self.face_dims(axis);
        if f.cmplt(IVec3::ZERO).any() || f.cmpge(dims.as_ivec3()).any() {
            return 0.0;
        }
        self.faces(axis)[(f.x as u32 + dims.x * (f.y as u32 + dims.y * f.z as u32)) as usize]
    }

    /// Make the water incompressible: solve the pressure Poisson equation
    /// on water cells (air has zero pressure, rock lets nothing through).
    fn project(&mut self, dt: f32, max_iterations: usize) {
        let n = self.n;
        let count = self.cells.len();
        // Water cells get an index in the system.
        let mut row = vec![usize::MAX; count];
        let mut water: Vec<IVec3> = Vec::new();
        for z in 0..n.z as i32 {
            for y in 0..n.y as i32 {
                for x in 0..n.x as i32 {
                    let i = idx(n, x as u32, y as u32, z as u32);
                    if self.cells[i] == Cell::Water {
                        row[i] = water.len();
                        water.push(IVec3::new(x, y, z));
                    }
                }
            }
        }
        if water.is_empty() {
            return;
        }
        let axes = [IVec3::X, IVec3::Y, IVec3::Z];
        // Right-hand side: minus the divergence (in units of h).
        let mut b = vec![0.0f32; water.len()];
        // Diagonal: number of non-solid neighbours.
        let mut diag = vec![0.0f32; water.len()];
        for (r, &c) in water.iter().enumerate() {
            let mut div = 0.0;
            let mut d = 0.0;
            for (axis, &e) in axes.iter().enumerate() {
                div += self.face_value(axis, c + e) - self.face_value(axis, c);
                for nb in [c - e, c + e] {
                    if !self.is_solid(nb) {
                        d += 1.0;
                    }
                }
            }
            b[r] = -div;
            diag[r] = d;
        }
        let neighbours = |r: usize| -> [Option<usize>; 6] {
            let c = water[r];
            let mut out = [None; 6];
            let mut k = 0;
            for e in axes {
                for nb in [c - e, c + e] {
                    if let Some(i) = self.cell_index(nb) {
                        if row[i] != usize::MAX {
                            out[k] = Some(row[i]);
                        }
                    }
                    k += 1;
                }
            }
            out
        };
        let nbs: Vec<[Option<usize>; 6]> = (0..water.len()).map(neighbours).collect();
        let apply = |x: &[f32], out: &mut [f32]| {
            for (r, o) in out.iter_mut().enumerate() {
                let mut s = diag[r] * x[r];
                for j in nbs[r].iter().flatten() {
                    s -= x[*j];
                }
                *o = s;
            }
        };
        // Conjugate gradients with a modified incomplete Cholesky
        // preconditioner (MIC(0)); rows run x fastest, then y, then z, so
        // neighbours 0, 2, 4 (-x, -y, -z) come before a row.
        let m = water.len();
        let has = |r: usize, k: usize| nbs[r][k].is_some() as u32 as f32;
        let mut precon = vec![0.0f32; m];
        const TAU: f32 = 0.97;
        const SIGMA: f32 = 0.25;
        for r in 0..m {
            let mut e = diag[r];
            for (k, up) in [(0usize, [3usize, 5usize]), (2, [1, 5]), (4, [1, 3])] {
                if let Some(q) = nbs[r][k] {
                    let pq = precon[q];
                    e -= pq * pq;
                    e -= TAU * (has(q, up[0]) + has(q, up[1])) * pq * pq;
                }
            }
            if e < SIGMA * diag[r] {
                e = diag[r];
            }
            precon[r] = if e > 0.0 { 1.0 / e.sqrt() } else { 0.0 };
        }
        let apply_precon = |r: &[f32], z: &mut [f32]| {
            // Forward: L q = r.
            let mut q = vec![0.0f32; m];
            for i in 0..m {
                let mut t = r[i];
                for k in [0usize, 2, 4] {
                    if let Some(j) = nbs[i][k] {
                        t += precon[j] * q[j];
                    }
                }
                q[i] = t * precon[i];
            }
            // Backward: L^T z = q.
            for i in (0..m).rev() {
                let mut t = q[i];
                for k in [1usize, 3, 5] {
                    if let Some(j) = nbs[i][k] {
                        t += precon[i] * z[j];
                    }
                }
                z[i] = t * precon[i];
            }
        };
        // Warm start from the last step's pressure (scaled to this step).
        let scale = if self.last_dt > 0.0 {
            dt / self.last_dt
        } else {
            0.0
        };
        let mut p: Vec<f32> = water
            .iter()
            .map(|c| self.pressure[idx(n, c.x as u32, c.y as u32, c.z as u32)] * scale)
            .collect();
        let mut r = vec![0.0f32; m];
        apply(&p, &mut r);
        for k in 0..m {
            r[k] = b[k] - r[k];
        }
        let mut z = vec![0.0f32; m];
        apply_precon(&r, &mut z);
        let mut s = z.clone();
        let mut rz: f32 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
        let tol = 1e-5 * b.iter().map(|x| x.abs()).fold(0.0, f32::max).max(1e-12);
        let mut q = vec![0.0f32; m];
        let mut it = 0;
        while it < max_iterations {
            if r.iter().map(|x| x.abs()).fold(0.0, f32::max) <= tol {
                break;
            }
            apply(&s, &mut q);
            let sq: f32 = s.iter().zip(&q).map(|(a, b)| a * b).sum();
            if sq.abs() < 1e-20 {
                break;
            }
            let alpha = rz / sq;
            for k in 0..m {
                p[k] += alpha * s[k];
                r[k] -= alpha * q[k];
            }
            apply_precon(&r, &mut z);
            let rz_new: f32 = r.iter().zip(&z).map(|(a, b)| a * b).sum();
            let beta = rz_new / rz.max(1e-30);
            rz = rz_new;
            for k in 0..m {
                s[k] = z[k] + beta * s[k];
            }
            it += 1;
        }
        self.last_dt = dt;
        self.last_iterations = it;
        self.pressure.iter_mut().for_each(|x| *x = 0.0);
        for (k, c) in water.iter().enumerate() {
            let i = idx(n, c.x as u32, c.y as u32, c.z as u32);
            self.pressure[i] = p[k];
        }
        // Subtract the gradient on every face next to water.
        for axis in 0..3 {
            let (dims, _) = self.face_dims(axis);
            let mut faces = std::mem::take(match axis {
                0 => &mut self.u,
                1 => &mut self.v,
                _ => &mut self.w,
            });
            for zf in 0..dims.z {
                for yf in 0..dims.y {
                    for xf in 0..dims.x {
                        let (a, bc) = self.face_cells(axis, UVec3::new(xf, yf, zf));
                        let (ka, kb) = (self.cell_kind(a), self.cell_kind(bc));
                        if ka != Cell::Water && kb != Cell::Water {
                            continue;
                        }
                        let j = (xf + dims.x * (yf + dims.y * zf)) as usize;
                        match (self.pressure_at(a), self.pressure_at(bc)) {
                            (Some(pa), Some(pb)) => faces[j] -= pb - pa,
                            _ => faces[j] = 0.0,
                        }
                    }
                }
            }
            match axis {
                0 => self.u = faces,
                1 => self.v = faces,
                _ => self.w = faces,
            }
        }
    }

    fn pressure_at(&self, c: IVec3) -> Option<f32> {
        match self.cell_index(c) {
            None => None,
            Some(i) => match self.cells[i] {
                Cell::Solid => None,
                Cell::Air => Some(0.0),
                Cell::Water => Some(self.pressure[i]),
            },
        }
    }

    /// Carry velocities a few cells into the air, so particles at the
    /// surface and in spray read sensible values.
    fn extrapolate(&mut self, layers: usize) {
        use rayon::prelude::*;
        let this = &*self;
        let mut out: Vec<Vec<f32>> = (0..3usize)
            .into_par_iter()
            .map(|axis| this.extrapolate_axis(axis, layers))
            .collect();
        self.w = out.pop().expect("3");
        self.v = out.pop().expect("3");
        self.u = out.pop().expect("3");
    }

    fn extrapolate_axis(&self, axis: usize, layers: usize) -> Vec<f32> {
        let (dims, _) = self.face_dims(axis);
        let len = (dims.x * dims.y * dims.z) as usize;
        let mut known = vec![false; len];
        for zf in 0..dims.z {
            for yf in 0..dims.y {
                for xf in 0..dims.x {
                    let (a, b) = self.face_cells(axis, UVec3::new(xf, yf, zf));
                    if self.cell_kind(a) == Cell::Water || self.cell_kind(b) == Cell::Water {
                        known[(xf + dims.x * (yf + dims.y * zf)) as usize] = true;
                    }
                }
            }
        }
        let mut data = self.faces(axis).clone();
        for _ in 0..layers {
            let mut next_known = known.clone();
            let mut next = data.clone();
            for zf in 0..dims.z as i32 {
                for yf in 0..dims.y as i32 {
                    for xf in 0..dims.x as i32 {
                        let j = (xf as u32 + dims.x * (yf as u32 + dims.y * zf as u32)) as usize;
                        if known[j] {
                            continue;
                        }
                        let mut s = 0.0;
                        let mut c = 0;
                        for e in [
                            IVec3::X,
                            IVec3::NEG_X,
                            IVec3::Y,
                            IVec3::NEG_Y,
                            IVec3::Z,
                            IVec3::NEG_Z,
                        ] {
                            let q = IVec3::new(xf, yf, zf) + e;
                            if q.cmplt(IVec3::ZERO).any() || q.cmpge(dims.as_ivec3()).any() {
                                continue;
                            }
                            let k =
                                (q.x as u32 + dims.x * (q.y as u32 + dims.y * q.z as u32)) as usize;
                            if known[k] {
                                s += data[k];
                                c += 1;
                            }
                        }
                        if c > 0 {
                            next[j] = s / c as f32;
                            next_known[j] = true;
                        }
                    }
                }
            }
            data = next;
            known = next_known;
        }
        data
    }

    fn grid_to_particles(&mut self) {
        use rayon::prelude::*;
        let this = &*self;
        let flip = self.flip;
        let vel: Vec<Vec3> = self
            .pos
            .par_iter()
            .zip(self.vel.par_iter())
            .map(|(&p, &v)| {
                let new = this.velocity_at(p);
                let old = this.old_velocity_at(p);
                (v + (new - old)) * flip + new * (1.0 - flip)
            })
            .collect();
        self.vel = vel;
    }

    fn advect(&mut self, dt: f32) {
        use rayon::prelude::*;
        let size = self.n.as_vec3() * self.h;
        let this = &*self;
        // None: the particle left the box.
        let moved: Vec<Option<(Vec3, Vec3)>> = self
            .pos
            .par_iter()
            .zip(self.vel.par_iter())
            .map(|(&p0, &vel)| {
                let v1 = this.velocity_at(p0);
                let mid = p0 + v1 * (0.5 * dt);
                let v2 = this.velocity_at(mid);
                let mut p = p0 + v2 * dt;
                if p.cmplt(Vec3::ZERO).any() || p.cmpge(size).any() {
                    return None;
                }
                let mut vel = vel;
                if this.is_solid(this.cell_of(p)) {
                    // Back out of the rock along the way it came, and lose
                    // the speed into it.
                    let mut lo = 0.0;
                    let mut hi = 1.0;
                    for _ in 0..6 {
                        let t = 0.5 * (lo + hi);
                        if this.is_solid(this.cell_of(p0.lerp(p, t))) {
                            hi = t;
                        } else {
                            lo = t;
                        }
                    }
                    p = p0.lerp(p, lo);
                    if this.is_solid(this.cell_of(p)) {
                        p = p0;
                    }
                    vel *= 0.5;
                }
                Some((p, vel))
            })
            .collect();
        let before = moved.len();
        self.pos.clear();
        self.vel.clear();
        for (p, v) in moved.into_iter().flatten() {
            self.pos.push(p);
            self.vel.push(v);
        }
        self.lost += before - self.pos.len();
    }

    /// One step of `dt` seconds (the caller keeps it under the CFL limit).
    pub fn step(&mut self, dt: f32, pressure_iterations: usize) {
        if self.pos.is_empty() {
            return;
        }
        self.classify();
        self.particles_to_grid();
        self.enforce_solids();
        self.u_old.clone_from(&self.u);
        self.v_old.clone_from(&self.v);
        self.w_old.clone_from(&self.w);
        self.apply_forces(dt);
        self.enforce_solids();
        self.project(dt, pressure_iterations);
        self.extrapolate(2);
        self.enforce_solids();
        self.grid_to_particles();
        self.advect(dt);
    }

    /// The largest particle speed.
    pub fn max_speed(&self) -> f32 {
        self.vel.iter().map(|v| v.length()).fold(0.0, f32::max)
    }
}
