import numpy as np
import pandas as pd
from typing import List
from datamate import root, Directory
from flyvis import renderings_dir
from flyvis.datasets.datasets import SequenceDataset
from flyvis.datasets.rendering import BoxEye
from tqdm import tqdm
import torch
        
class Glider1D:
    def __init__(
        self,
        rule,
        parity,
        x_resol=33,
        ysize=None,
        T=72,
        vel=1,
        seed=None,
        direction="pd",  # "pd" (preferred) or "nd" (null)
        orientation="x", # <--- NEW: Add orientation parameter
    ):
        # --- CHANGE 1: Add 'uniform_edge' to the list of valid rules ---
        assert rule in {"2pt", "3pt_Div", "3pt_Conv", "uniform_edge", 'uncorrelated'}
        assert parity in (+1, -1)
        assert direction in {"pd", "nd"}
        assert orientation in {"x", "y"} # <--- NEW: Validate orientation
        
        self.rule = rule
        self.parity = np.int8(parity)
        self.x_resol = int(x_resol)
        self.ysize = int(ysize) if ysize is not None else int(x_resol)
        self.T = int(T)
        self.vel = int(vel)
        self.direction = direction
        self.orientation = orientation # <--- NEW: Store orientation
        self.rng = np.random.default_rng(seed)

        self._Sxt = None        # (spatial_resol, T)
        self._init_col = None     # S[:,0]
        self._left_edge = None    # S[0,:]

    # ---------- public API ----------
    
    @property
    def _spatial_resol(self):
        """Helper property to get the active spatial dimension."""
        if self.orientation == "x":
            return self.x_resol
        else:
            return self.ysize

    def evolve(self, init_col=None, left_edge=None):
        """Fill lattice (spatial_resol, T) in-place using the chosen rule and boundaries."""
        
        # <--- CHANGED: Use _spatial_resol
        Sxt = np.empty((self._spatial_resol, self.T), dtype=np.int8)

        # --- CHANGE 2: Add a special case to generate the uniform edge directly ---
        if self.rule == "uniform_edge":
            Sxt = self._generate_uniform_edge()
        elif self.rule == 'uncorrelated':
            Sxt = self._uncorrelated()
        else:
            # This is the original evolution logic for other rules
            if init_col is None:
                # <--- CHANGED: Use _spatial_resol
                init_col = self._rand_pm1(self._spatial_resol)
            if left_edge is None:
                left_edge = self._rand_pm1(self.T)

            Sxt[:, 0] = init_col
            Sxt[0, :] = left_edge
            self._init_col = init_col
            self._left_edge = left_edge

            for t in range(self.T - 1):
                if self.rule == "3pt_Conv":
                    self._step_inv_L(Sxt, t)
                elif self.rule == "3pt_Div":
                    self._step_L(Sxt, t)
                elif self.rule == "2pt":
                    self._step_motion(Sxt, t)

        # ---- Apply directionality (preferred vs null) ----
        if self.direction == "nd":
            Sxt = Sxt[::-1, :]

        # store & sanity check
        self._Sxt = Sxt
        if not np.all(np.isin(Sxt, (-1, 1))):
            raise RuntimeError("Non {±1} spin detected—check boundaries/rule.")
        return Sxt

    def _generate_uniform_edge(self):
        """Generates a uniform edge moving at a constant velocity."""
        
        # <--- CHANGED: Use _spatial_resol
        # Create coordinate grids for space (x or y) and t (time)
        coords = np.arange(self._spatial_resol)[:, None]  # Shape: (spatial_resol, 1)
        
        # <--- CHANGED: Use _spatial_resol
        edge_positions = (np.arange(self.T)[None, :] * self.vel) % self._spatial_resol

        # Create the edge matrix using broadcasting.
        # The condition `coords >= edge_positions` becomes a boolean (spatial_resol, T) matrix.
        # np.where converts this to -1 and +1 based on the parity.
        
        # <--- CHANGED: Use coords
        Sxt = np.where(coords >= edge_positions, -self.parity, self.parity)
        return Sxt.astype(np.int8)

    def get_Sxt(self):
        self._require_evolved()
        return self._Sxt

    def to_S(self, broadcast=True, copy=False):
        """
        Converts the 1D pattern (_Sxt) to the full 3D stimulus (x, y, T),
        broadcasting along the correct axis based on self.orientation.
        """
        self._require_evolved()
        Sxt = self._Sxt # Shape: (spatial_resol, T)
        final_shape = (self.x_resol, self.ysize, self.T)

        # <--- NEW: Conditional broadcasting logic
        if self.orientation == "x":
            # Sxt shape is (x_resol, T). Broadcast along axis 1 (y).
            S_intermediate = Sxt[:, None, :] # Shape: (x_resol, 1, T)
            repeat_axis = 1
            repeats = self.ysize
        else: # self.orientation == "y"
            # Sxt shape is (ysize, T). Broadcast along axis 0 (x).
            S_intermediate = Sxt[None, :, :] # Shape: (1, ysize, T)
            repeat_axis = 0
            repeats = self.x_resol

        if broadcast:
            S = np.broadcast_to(S_intermediate, final_shape)
            if copy:
                S = np.ascontiguousarray(S)
            return S
        else:
            return np.repeat(S_intermediate, repeats, axis=repeat_axis)

    def to_flyvis(self, map01=True, dtype=np.float32):
        # This method requires no changes, as to_S() now correctly
        # returns the (x, y, T) array regardless of orientation.
        S = self.to_S(broadcast=True, copy=False)
        if map01:
            S = (S + 1) / 2.0
        S_tyx = np.transpose(S, (2, 1, 0))
        return np.asarray(S_tyx, dtype=dtype)[None, ...]

    def _step_inv_L(self, Sxt, t):
        Sxt[1:, t+1] = self.parity * (1 // (Sxt[:-1, t] * Sxt[1:, t]))

    def _step_L(self, Sxt, t):
        Sxt[1:, t+1] = Sxt[0, t+1] * np.cumprod(self.parity * Sxt[:-1, t], axis=0)

    def _step_motion(self, Sxt, t):
        Sxt[:, t+1] = self.parity * np.roll(Sxt[:, t], shift=self.vel, axis=0)

    def _uncorrelated(self):
        """Generates uncorrelated ±1 stimulus with parity-controlled contrast."""
        
        # <--- CHANGED: Use _spatial_resol
        # independent ±1 per (spatial_resol, t)
        Sxt = self._rand_pm1(self._spatial_resol * self.T).reshape(self._spatial_resol, self.T)
        
        # If parity is -1, perform an in-place multiplication to flip the contrast.
        if self.parity == -1:
            print("Inverting contrast for uncorrelated stimulus.")
            Sxt *= -1
        return Sxt

    def _rand_pm1(self, n):
        return self.rng.choice(np.array([-1, 1], dtype=np.int8), size=n, replace=True)

    def _require_evolved(self):
        if self._Sxt is None:
            raise RuntimeError("Call evolve() first.")

# class Glider1D:
#     def __init__(
#         self,
#         rule,
#         parity,
#         x_resol=33,
#         ysize=None,
#         T=72,
#         # --- CHANGED: vel is now optional, only for uniform_edge ---
#         vel=None, 
#         seed=None,
#         direction="pd",
#         orientation="x",
#         fps=40.0,
#         dpp=2.25
#     ):
#         assert rule in {"2pt", "3pt_Div", "3pt_Conv", "uniform_edge", 'uncorrelated'}
#         assert parity in (+1, -1)
#         assert direction in {"pd", "nd"}
#         assert orientation in {"x", "y"}
        
#         self.rule = rule
        
#         # --- NEW: Check vel parameter ---
#         if self.rule == "uniform_edge":
#             if vel is None:
#                 raise ValueError("Must provide 'vel' (in deg/s) for 'uniform_edge' rule.")
#         elif vel is not None:
#             print(f"Warning: 'vel' parameter is ignored for rule '{self.rule}'.")

#         self.parity = np.int8(parity)
#         self.x_resol = int(x_resol)
#         self.ysize = int(ysize) if ysize is not None else int(x_resol)
#         self.T = int(T)
#         self.vel = vel # Stored in deg/s if rule is 'uniform_edge', else None
#         self.direction = direction
#         self.orientation = orientation
#         self.rng = np.random.default_rng(seed)
        
#         self.fps = fps
#         self.dpp = dpp

#         self._Sxt = None
#         self._init_col = None
#         self._left_edge = None

#     @property
#     def _spatial_resol(self):
#         """Helper property to get the active spatial dimension."""
#         if self.orientation == "x":
#             return self.x_resol
#         else:
#             return self.ysize

#     def evolve(self, init_col=None, left_edge=None):
#         """Fill lattice (spatial_resol, T) in-place using the chosen rule and boundaries."""
        
#         Sxt = np.empty((self._spatial_resol, self.T), dtype=np.int8)

#         if self.rule == "uniform_edge":
#             Sxt = self._generate_uniform_edge()
#         elif self.rule == 'uncorrelated':
#             Sxt = self._uncorrelated()
#         else:
#             # This is the original evolution logic for other rules
#             if init_col is None:
#                 init_col = self._rand_pm1(self._spatial_resol)
#             if left_edge is None:
#                 left_edge = self._rand_pm1(self.T)

#             Sxt[:, 0] = init_col
#             Sxt[0, :] = left_edge
#             self._init_col = init_col
#             self._left_edge = left_edge

#             for t in range(self.T - 1):
#                 if self.rule == "3pt_Conv":
#                     self._step_inv_L(Sxt, t)
#                 elif self.rule == "3pt_Div":
#                     self._step_L(Sxt, t)
#                 elif self.rule == "2pt":
#                     self._step_motion(Sxt, t)

#         # ---- Apply directionality (preferred vs null) ----
#         if self.direction == "nd":
#             Sxt = Sxt[::-1, :]

#         # store & sanity check
#         self._Sxt = Sxt
#         if not np.all(np.isin(Sxt, (-1, 1))):
#             raise RuntimeError("Non {±1} spin detected—check boundaries/rule.")
#         return Sxt

#     def _generate_uniform_edge(self):
#         """
#         Generates a uniform edge moving at a constant velocity.
#         self.vel is interpreted as DEGREES PER SECOND.
#         """
        
#         # 1. Convert velocity from deg/s to px/frame
#         vel_px_s = self.vel / self.dpp       # (deg/s) / (deg/px) = px/s
#         vel_px_frame = vel_px_s / self.fps   # (px/s) / (frames/s) = px/frame
        
#         # 2. Create coordinate grids
#         coords = np.arange(self._spatial_resol)[:, None]  # Shape: (spatial_resol, 1)
#         time_frames = np.arange(self.T)[None, :]         # Shape: (1, T)
        
#         # 3. Calculate edge position at each frame
#         edge_positions = (time_frames * vel_px_frame) % self._spatial_resol

#         # 4. Create the kymograph
#         Sxt = np.where(coords >= edge_positions, -self.parity, self.parity)
#         return Sxt.astype(np.int8)

#     def get_Sxt(self):
#         self._require_evolved()
#         return self._Sxt

#     def to_S(self, broadcast=True, copy=False):
#         """
#         Converts the 1D pattern (_Sxt) to the full 3D stimulus (x, y, T),
#         broadcasting along the correct axis based on self.orientation.
#         """
#         self._require_evolved()
#         Sxt = self._Sxt # Shape: (spatial_resol, T)
#         final_shape = (self.x_resol, self.ysize, self.T)

#         if self.orientation == "x":
#             S_intermediate = Sxt[:, None, :] # Shape: (x_resol, 1, T)
#             repeat_axis = 1
#             repeats = self.ysize
#         else: # self.orientation == "y"
#             S_intermediate = Sxt[None, :, :] # Shape: (1, ysize, T)
#             repeat_axis = 0
#             repeats = self.x_resol

#         if broadcast:
#             S = np.broadcast_to(S_intermediate, final_shape)
#             if copy:
#                 S = np.ascontiguousarray(S)
#             return S
#         else:
#             return np.repeat(S_intermediate, repeats, axis=repeat_axis)

#     def to_flyvis(self, map01=True, dtype=np.float32):
#         S = self.to_S(broadcast=True, copy=False)
#         if map01:
#             S = (S + 1) / 2.0
#         S_tyx = np.transpose(S, (2, 1, 0))
#         return np.asarray(S_tyx, dtype=dtype)[None, ...]

#     def _step_inv_L(self, Sxt, t):
#         Sxt[1:, t+1] = self.parity * (1 // (Sxt[:-1, t] * Sxt[1:, t]))

#     def _step_L(self, Sxt, t):
#         Sxt[1:, t+1] = Sxt[0, t+1] * np.cumprod(self.parity * Sxt[:-1, t], axis=0)

#     def _step_motion(self, Sxt, t):
#         # --- CHANGED: Hardcoded shift to 1 px/frame ---
#         Sxt[:, t+1] = self.parity * np.roll(Sxt[:, t], shift=1, axis=0)

#     def _uncorrelated(self):
#         """Generates uncorrelated ±1 stimulus with parity-controlled contrast."""
        
#         Sxt = self._rand_pm1(self._spatial_resol * self.T).reshape(self._spatial_resol, self.T)
        
#         if self.parity == -1:
#             print("Inverting contrast for uncorrelated stimulus.")
#             Sxt *= -1
#         return Sxt

#     def _rand_pm1(self, n):
#         return self.rng.choice(np.array([-1, 1], dtype=np.int8), size=n, replace=True)

#     def _require_evolved(self):
#         if self._Sxt is None:
#             raise RuntimeError("Call evolve() first.")


# root tells where the Directory-tree starts
@root(renderings_dir)
class RenderedData(Directory):
    class Config(dict):
        extent: int              # radius, in number of receptors of the hexagonal array
        kernel_size: int         # photon collection radius, in pixels
        subset_idx: List[int]    # if specified, subset of sequences to render
        x_resol: int             # horizontal resolution of the glider pattern
        T: int                   # number of frames in the glider pattern
        rules: List[str]         # glider rules, e.g. ["motion_1d", "L", "inv_L"]
        directions: List[str]    # ["pd", "nd"]
        parities: List[int]              # +1 or -1
        orientations: List[str]
        vel: int                 # velocity of the glider
        seed: int                # RNG seed

    def __init__(self, config: Config):
        # 1) Build specs for all requested gliders
        glider_specs = []
        for rule in config.rules:
            for direction in config.directions:
                for parity in config.parities:
                    for orientation in config.orientations: 
                        is_motion_rule = rule in ["2pt", "uniform_edge"]
                        glider_specs.append({
                            "rule": rule,
                            "direction": direction,
                            "parity" : parity, 
                            "orientation": orientation, # <--- 3. ADD THIS
                            "vel": config.vel if is_motion_rule else 0,
                        })

        seqs = []
        meta = []
        # 2) Generate sequences
        for spec in glider_specs:
            g = Glider1D(
                x_resol=config.x_resol,
                T=config.T,
                ysize=config.x_resol,
                rule=spec["rule"],
                parity=spec["parity"],
                vel=spec["vel"],
                seed=config.seed,
                direction=spec["direction"],
                orientation = spec['orientation'],
            )
            g.evolve()
            seq = g.to_flyvis()    # (1, T, H, W)
            seqs.append(seq)
            meta.append(spec)

        sequences = np.concatenate(seqs, axis=0)   # (N, T, H, W)

        # 3) Render (batch, frames, H, W) -> (batch, frames, 1, hexals)
        receptors = BoxEye(extent=config.extent, kernel_size=config.kernel_size)
        subset_idx = getattr(config, "subset_idx", []) or list(range(sequences.shape[0]))

        rendered_sequences = []
        with tqdm(total=len(subset_idx), desc="Rendering glider → hex") as pbar:
            for index in subset_idx:
                rendered = receptors(sequences[[index]]).cpu().numpy()
                rendered_sequences.append(rendered)
                pbar.update()

        rendered_sequences = np.concatenate(rendered_sequences, axis=0)  # (N, T, 1, hexals)

        # 4) Save to disk via Directory protocol
        self.sequences = rendered_sequences    # (N, T, 1, hexals)
        self.cartesian_sequences = sequences   # (N, T, H, W)
    
class CustomStimuli(SequenceDataset):
    """Custom stimuli dataset that can resample sequences based on dt and original framerate."""

    def __init__(self, rendered_data_config: dict, dt: float, original_framerate: float):
        super().__init__()
        self.dt = dt
        self.original_framerate = original_framerate
        self.t_pre = 0.5
        self.t_post = 0.5
        self.augment = False

        # load sequences from rendered data
        self.dir = RenderedData(rendered_data_config)
        self.sequences = torch.tensor(self.dir.sequences[:])
        self.cartesian_sequences = self.dir.cartesian_sequences[:]
        self.n_sequences = self.sequences.shape[0]

        # rebuild metadata from config
        rules = rendered_data_config["rules"]
        directions = rendered_data_config["directions"]
        parities = rendered_data_config["parities"]
        orientations = rendered_data_config["orientations"] # <--- 1. ADD THIS
        vel = rendered_data_config["vel"]

        glider_meta = []
        for rule in rules:
            for direction in directions:
                for parity in parities:
                    # --- 2. ADD THIS LOOP ---
                    for orientation in orientations:
                        # --- 3. FIX BUG for consistency ---
                        is_motion_rule = rule in ["2pt", "uniform_edge"] 
                        glider_meta.append({
                            "rule": rule,
                            "direction": direction,
                            "parity": parity,
                            "orientation": orientation, # <--- 4. ADD THIS
                            "vel": vel if is_motion_rule else 0, # <--- 3. (FIXED)
                        })

        # subset selection
        subset_idx = rendered_data_config.get("subset_idx", [])
        if not subset_idx:  # empty means "all"
            subset_idx = list(range(len(glider_meta)))
        glider_meta = [glider_meta[i] for i in subset_idx]

        # argument dataframe
        self.arg_df = pd.DataFrame(glider_meta)
        self.arg_df.insert(0, "sequence_idx", np.arange(self.n_sequences))

    def get_item(self, key):
        sequence = self.sequences[key]
        # resample to match desired temporal resolution
        resample = self.get_temporal_sample_indices(
            sequence.shape[0],
            sequence.shape[0]
        )
        return sequence[resample]

# # root tells where the Directory-tree starts
# @root(renderings_dir)
# class RenderedData(Directory):
#     class Config(dict):
#         extent: int             # radius, in number of receptors
#         kernel_size: int        # photon collection radius, in pixels
#         subset_idx: List[int]   # if specified, subset of sequences to render
#         x_resol: int            # horizontal resolution of the glider pattern
#         T: int                  # number of frames in the glider pattern
#         rules: List[str]        # glider rules
#         directions: List[str]   # ["pd", "nd"]
#         parities: List[int]     # +1 or -1
#         orientations: List[str]
#         vel: int                # velocity of the glider (in deg/s, for uniform_edge)
#         seed: int               # RNG seed
#         # --- NEW: Added fps and dpp ---
#         fps: float              # frames per second
#         dpp: float              # degrees per pixel

#     def __init__(self, config: Config):
#         # 1) Build specs for all requested gliders
#         glider_specs = []
#         for rule in config.rules:
#             for direction in config.directions:
#                 for parity in config.parities:
#                     for orientation in config.orientations: 
                        
#                         # --- CHANGED: vel is now specific to uniform_edge ---
#                         spec_vel = None
#                         if rule == "uniform_edge":
#                             spec_vel = config.vel
                        
#                         glider_specs.append({
#                             "rule": rule,
#                             "direction": direction,
#                             "parity" : parity, 
#                             "orientation": orientation,
#                             "vel": spec_vel, # --- CHANGED: Pass specific vel
#                         })

#         seqs = []
#         meta = []
#         # 2) Generate sequences
#         for spec in glider_specs:
#             g = Glider1D(
#                 x_resol=config.x_resol,
#                 T=config.T,
#                 ysize=config.x_resol,
#                 rule=spec["rule"],
#                 parity=spec["parity"],
#                 vel=spec["vel"],        # --- CHANGED: Pass correct vel (None or deg/s)
#                 seed=config.seed,
#                 direction=spec["direction"],
#                 orientation = spec['orientation'],
#                 # --- NEW: Pass fps and dpp ---
#                 fps=config.fps,
#                 dpp=config.dpp
#             )
#             g.evolve()
#             seq = g.to_flyvis()     # (1, T, H, W)
#             seqs.append(seq)
#             meta.append(spec)

#         sequences = np.concatenate(seqs, axis=0)   # (N, T, H, W)

#         # ... (rest of the class is unchanged) ...
        
#         # 3) Render (batch, frames, H, W) -> (batch, frames, 1, hexals)
#         # (Assuming BoxEye is defined elsewhere)
#         receptors = BoxEye(extent=config.extent, kernel_size=config.kernel_size) 
#         subset_idx = getattr(config, "subset_idx", []) or list(range(sequences.shape[0]))

#         rendered_sequences = []
#         with tqdm(total=len(subset_idx), desc="Rendering glider → hex") as pbar:
#             for index in subset_idx:
#                 rendered = receptors(sequences[[index]]).cpu().numpy()
#                 rendered_sequences.append(rendered)
#                 pbar.update()

#         rendered_sequences = np.concatenate(rendered_sequences, axis=0)   # (N, T, 1, hexals)

#         # 4) Save to disk via Directory protocol
#         self.sequences = rendered_sequences      # (N, T, 1, hexals)
#         self.cartesian_sequences = sequences   # (N, T, H, W)

# class CustomStimuli(SequenceDataset):
#     """Custom stimuli dataset that can resample sequences based on dt and original framerate."""

#     # --- CHANGED: Added dpp ---
#     def __init__(self, rendered_data_config: dict, dt: float, original_framerate: float, dpp: float):
#         super().__init__()
#         self.dt = dt
#         self.original_framerate = original_framerate
#         self.dpp = dpp # --- NEW ---
#         self.t_pre = 0.5
#         self.t_post = 0.5
#         self.augment = False

#         # --- NEW: Add fps and dpp to the config before passing it ---
#         rendered_data_config["fps"] = self.original_framerate
#         rendered_data_config["dpp"] = self.dpp

#         # load sequences from rendered data
#         self.dir = RenderedData(rendered_data_config)
#         self.sequences = torch.tensor(self.dir.sequences[:])
#         self.cartesian_sequences = self.dir.cartesian_sequences[:]
#         self.n_sequences = self.sequences.shape[0]

#         # rebuild metadata from config
#         rules = rendered_data_config["rules"]
#         directions = rendered_data_config["directions"]
#         parities = rendered_data_config["parities"]
#         orientations = rendered_data_config["orientations"]
#         # --- CHANGED: Use .get() for vel, it might not exist ---
#         vel = rendered_data_config.get("vel")

#         glider_meta = []
#         for rule in rules:
#             for direction in directions:
#                 for parity in parities:
#                     for orientation in orientations:
                        
#                         # --- CHANGED: Logic for storing vel metadata ---
#                         spec_vel = None
#                         if rule == "uniform_edge":
#                             spec_vel = vel
                        
#                         glider_meta.append({
#                             "rule": rule,
#                             "direction": direction,
#                             "parity": parity,
#                             "orientation": orientation,
#                             "vel": spec_vel, # --- CHANGED ---
#                         })

#         # subset selection
#         subset_idx = rendered_data_config.get("subset_idx", [])
#         if not subset_idx:  # empty means "all"
#             subset_idx = list(range(len(glider_meta)))
#         glider_meta = [glider_meta[i] for i in subset_idx]

#         # argument dataframe
#         self.arg_df = pd.DataFrame(glider_meta)
#         self.arg_df.insert(0, "sequence_idx", np.arange(self.n_sequences))

#     def get_item(self, key):
#         sequence = self.sequences[key]
#         # resample to match desired temporal resolution
#         resample = self.get_temporal_sample_indices(
#             sequence.shape[0],
#             sequence.shape[0]
#         )
#         return sequence[resample]