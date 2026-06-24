
# vim: tabstop=4 shiftwidth=4 expandtab
"""
Class for lpdm inversions
"""
from __future__ import print_function

import os, pdb
import sys
import datetime
import shutil
import math
import numpy
import netCDF4
from scipy.sparse import csr_matrix, save_npz, load_npz

from configobj import ConfigObj

ONEDAY = datetime.timedelta(days=1)

class lpdm:
    """ Class for lpdm inversions.

    Settings for this class to use are in a configuration file,
    which is passed into the class.  All of the entries in the config file
    are converted to class attributes with the same name.

    Parameters
    ----------
    configfile : str
        Name of configuration file.  Default is 'config.ini' in the current
        working directory.

    debug : bool
        If true, print out extra debugging information during processing.

    See Also
    --------
    lpdm programs that use this class:
        hsplit.py, hsigma.py, hq.py, make_z.py, inversion.py

    ConfigObj - module used for reading configuration file.

    Example
    -------
    >>> ctl = lpdm.lpdm("config.ini")
    >>> foot, dates, lats, lons = ctl.get_footprint(filename)
    """


    def __init__(self, configfile="config.ini", debug=False):

        if not os.path.exists(configfile):
            sys.exit("Error: config file '%s' not found." % configfile)

        self.debug = debug

        config = ConfigObj(configfile, unrepr=True)
        if not self._check_config(config):
            sys.exit()

        # simple conversion of config dict to attributes.
        # does not work if dict item is itself a dict
        for a, b in config.items():
            if self.debug:
                print("Creating attribute %s with value %s" % (a, b))
            setattr(self, a, b)


        self.steps_per_day = 24.0/self.hrsperstep
        self.timestep = datetime.timedelta(hours=self.hrsperstep)

        # Get start and end dates for the inversion
        (syr, smon, sday) = self.start_date.split("-")
        (eyr, emon, eday) = self.end_date.split("-")
        self.sdate = datetime.datetime(int(syr), int(smon), int(sday))
        self.edate = datetime.datetime(int(eyr), int(emon), int(eday))
        self.edate += ONEDAY
        self.ndays = (self.edate-self.sdate).days

        self.ntimesteps = int(self.ndays * self.steps_per_day)
        self.timestep = datetime.timedelta(hours=self.hrsperstep)

        self.nlatgrids = round((self.north - self.south) / self.lat_resolution) + 1
        self.nlongrids = round((self.east - self.west) / self.lon_resolution) + 1
        self.ncells = self.nlatgrids * self.nlongrids

        self.landmask = self._get_landmask(self.workdir, self.landmask_file)
        self.nlandcells = len(self.landmask[0])

        # convert indices of land cells into actual latitudes and longitudes
        #self.lats = self.landmask[0] + self.south + self.lat_resolution
        #self.lons = self.landmask[1] + self.west + self.lon_resolution
        all_lats = numpy.arange(self.south,self.north+self.lat_resolution/2,self.lat_resolution)
        all_lons = numpy.arange(self.west,self.east+self.lon_resolution/2,self.lon_resolution)
        self.lats = all_lats[self.landmask[0]]
        self.lons = all_lons[self.landmask[1]]
        
        # set directory paths
        self.tmpdir = "/".join([self.workdir, self.h_tmp_dir])
        self.full_hdir = "/".join([self.workdir, self.hdir])
        self.hq_dir = "/".join([self.workdir, self.hqdir])
        self.hsig_dir = "/".join([self.workdir, self.hsigdir])

    #--------------------------------------------------------------
    @staticmethod
    def _check_config(config):
        """ Check that required entries in the config file exist """

        required = ['hrsperstep',
                    'workdir',
                    'start_date',
                    'end_date',
                    'north',
                    'south',
                    'east',
                    'west',
                    'lat_resolution',
                    'lon_resolution',
                    'landmask_file',
                    'h_tmp_dir',
                    'hdir',
                    'hqdir',
                    'hsigdir',
                    'temporal_corr_length',
                    'sp_cov_file'
                    ]

        valid = True
        for name in required:
            if name not in config:
                print("Config file missing '%s' entry." % name)
                valid = False

        return valid

    #--------------------------------------------------------------
    def _get_landmask(self, dirname, filename):
        """ Read in the landmask file specified in the configuration file.

        Parameters
        ----------
        dirname : str
            Directory name where landmask file resides.
        filename : str
            File name of landmask file.
        """


        landmaskfile = dirname + "/" + filename
        if self.debug:
            print("Attempt to read landmask file %s" % landmaskfile)
        if os.path.exists(landmaskfile):
            landmaparr = numpy.load(landmaskfile)
        else:
            sys.exit("Landmask file %s not found." % (landmaskfile))

        return landmaparr

    #--------------------------------------------------------------
    def getCellLatLon(self, cellnum):
        """ Given a land cell number, return its latitude and longitude

        Parameters
        ----------
        cellnum : int
            cell number

        Returns
        -------
        lat, lon : float
            latitude and longitude of cell
        """

        if cellnum < 0 or cellnum >= len(self.lats):
            print("In getCellLatLon, cellnum out of range (%d). Should be > 0 and < %d"
                  % (cellnum, len(self.lats)), file=sys.stderr)
            return -999, -999

        lat = self.lats[cellnum]
        lon = self.lons[cellnum]

        return lat, lon

    #--------------------------------------------------------------
    def getCellNum(self, lat, lon):
        """ Given a latitidue and longitude, find the corresponding land cell number

        Parameters
        ----------
        lat, lon : float
            latitude and longitude

        Returns
        -------
        n : int
            cell number. -1 if lat and lon are not in domain
        """

        for n, (lt, ln) in enumerate(zip(self.lats, self.lons)):
            if lat == lt and lon == ln:
                return n

        return -1

    #--------------------------------------------------------------
    def prepare_H(self):
        """ Prepare directories to hold temporary H slice files.

        Creates the 'h_tmp_dir' and 'hdir' directories specified in the
        config file, or removes all files in the directories if they exist.

        Returns
        -------
        tmpdir, hdir : str
            full path to temporary H directory and H directory
        """

        # Temporary directory to hold text files
        if os.path.exists(self.tmpdir):
            if self.debug:
                print("Removing files in %s" % self.tmpdir)
            # remove any files in tmpdir
            shutil.rmtree(self.tmpdir)

        # create tmpdir
        if self.debug:
            print("Create directory %s" % self.tmpdir)

        try:
            os.makedirs(self.tmpdir)
        except Exception as e:
            sys.exit("Can't make directory for h files: %s. %s" % (self.tmpdir, e))

        # set the directory where the H slices will be stored.
        if os.path.exists(self.full_hdir):
            if self.debug:
                print("Removing files in %s" % self.full_hdir)
            # remove any files in the hdir.
            shutil.rmtree(self.full_hdir)

        # create the h directory
        if self.debug:
            print("Create directory %s" % self.full_hdir)
        try:
            os.makedirs(self.full_hdir)
        except Exception as e:
            sys.exit("Can't make directory for h files: %s. %s" % (self.full_hdir, e), file=sys.stderr)

        return self.tmpdir, self.full_hdir

    #--------------------------------------------------------------
    def prepare_Hsigma(self):
        """ Prepare directory to hold H*sigma files.

        Creates the 'hsigdir' directory specified in the
        config file, or removes all files in the directory if it exists.

        Returns
        -------
        hsig_dir : str
            full path to Hsigma directory.
        """

        # set the directory where the Hsigma slices will be stored.
        if os.path.exists(self.hsig_dir):
            # remove any files in the hdir.
            shutil.rmtree(self.hsig_dir)

        # create the h directory
        try:
            os.makedirs(self.hsig_dir)
        except Exception as e:
            sys.exit("Can't make directory for hsigma files: %s %s" % (self.hsig_dir, e), file=sys.stderr)

        return self.hsig_dir

    #--------------------------------------------------------------
    def prepare_HQ(self):
        """ Prepare directory to hold H*Q files.

        This routine will use an existing directory if present, or create
        it if not.  It does not remove any files in an existing directory.
        """

        if not os.path.exists(self.hq_dir):
            try:
                os.makedirs(self.hq_dir)
            except Exception as e:
                sys.exit("Can't make directory for hqsub files: %s %s" % (self.hq_dir, e), file=sys.stderr)

    #--------------------------------------------------------------
    def get_tmp_h_file(self, n):
        """ Determine the name of the temporary file that holds the h data.

        The directory name is set with 'h_tmp_dir' in the config file.

        Parameters
        ----------
        n : int
            timestep number

        Returns
        -------
        tmpfile : str
            Full path and file name of temporary H strip file
        """

        tmpfile = "%s/H%04d.txt" % (self.tmpdir, n)

        return tmpfile

    #--------------------------------------------------------------
    def get_h_file(self, n):
        """ Get the file name with the H strip data.

        The directory name is set with 'hdir' in the config file.

        Parameters
        ----------
        n : int
            The timestep number

        Returns
        -------
        hfile : str
            Full path and file name of H strip file
        """

        hfile = "%s/H%04d.npz" % (self.full_hdir, n)

        return hfile


    #--------------------------------------------------------------
    def get_hq_file(self, n):
        """ Get the name of the hq file for timestep n

        The directory name is set with 'hqdir' in the config file.

        Parameters
        ----------
        n : int
            timestep number

        Returns
        -------
        filename : str
            Full path and name of HQ file.
        """

        filename = self.hq_dir + "/HQ%04d" % (n)

        return filename

    #--------------------------------------------------------------
    def convert_h_file(self, n, num_obs):
        """ Convert the temporary H text files to numpy savez
        sparse matrix format for the given timestep number n.

        Location of temporary H text files is set by 'h_tmp_dir'
        setting in the config file, and the location of the
        H files is set by the 'hdir' setting in the config file.

        Parameters
        ----------
        n : int
            timestep number

        num_obs : int
            The total number of receptors, needed for the
            sparse format that the files are saved as.

        These files can be read using the read_sparse_h() method.
        """

        tmpfile = self.get_tmp_h_file(n)
        filename = self.get_h_file(n)

        print("Convert ", tmpfile, "to", filename)

        dtype = [('obs', numpy.int32), ('cell', numpy.int32), ('val', float)]
        if os.path.exists(tmpfile):
            h = numpy.loadtxt(tmpfile, dtype=dtype)
            if h.size == 1:  # handle the strange way numpy treats single row files
                h = h.reshape(1)
        else:
            h = numpy.array([(0, 0, 0)], dtype=dtype)
            print("WARNING: Text file %s does not exist.  Creating empty H file" % tmpfile)

        #               data       row       col
        m = csr_matrix((h['val'], (h['obs'], h['cell'])), shape=(num_obs, self.nlandcells))
        save_npz(filename, m)

    #--------------------------------------------------------------
    def write_sparse(self, filename, matr):
        """ Save a sparse matrix to file.

        Parameters
        ----------
        filename : str
            The name of the file to write to
        matr : array
            A scipy sparse matrix
        """

        save_npz(filename, matr)

    #--------------------------------------------------------------
    def read_sparse_h(self, timestep, getSparse=True, hdir=None):
        """ Read a sparse h slice file and return a sparse matrix.

        Parameters
        ----------
        timestep : int
            time step number
        getSparse : bool, optional
            If true, get a sparse matrix from data file. If false,
            get full matrix from data file.
        hdir : str, optional
            Directory where H slice files are located.  If not set,
            use 'hdir' directory specified in config file.

        Returns
        -------
        data : array
            csr sparse matrix with data, if getSparse is True, or
            full array (populated with 0 where not specified) if getSparse is False.
        """

        if hdir is None:
            Hdir = self.full_hdir
        else:
            Hdir = hdir

        hfile = Hdir + "/H%04d.npz" % (timestep+1)

        if os.path.exists(hfile):
            h = load_npz(hfile)
            if not getSparse:
                h = h.todense()
        else:
            sys.exit("H file does not exist for step %d" % timestep)

        return h

    #--------------------------------------------------------------
    def get_footprint(self, filename, extract_domain=False):
        """ Return footprint from netcdf file.

        This is the netcdf variable 'foot1' from the stilt netcdf footprint files.

        Parameters
        ----------
        filename : str
            The name of the netcdf file

        extract_domain : str, optional
            If true, extract the inversion grid domain from the footprint grid.

        Returns
        -------
        grid : array
            The footprint grid, or subgrid if extract_domain is True
        dates : array
            Dates for the grid
        lats : array
            Latitudes of footprint grid (not the subgrid if extract_domain is True)
        lons : array
            Longitudes of footprint grid (not the subgrid if extract_domain is True)
        """

        try:
            ds = netCDF4.Dataset(filename)
        except IOError as e:
            print("Error trying to read netcdf file %s. %s" % (filename, e), file=sys.stderr)
            return None, None, None, None

        g = ds.variables['foot']
        grid = g[:]
        lats = ds.variables['lat'][:]
        lons = ds.variables['lon'][:]
#        dy = lats[1] - lats[0]
        dx = lons[1] - lons[0]

        s = list(g.dimensions)
        n1 = s.index("lat")
        n2 = s.index("lon")

        # Convert the foot1date array from days since jan 1, 2000 to actual datetime
        f1 = ds.variables['time'][:]
        dates = numpy.array([datetime.datetime(1970,1,1) + datetime.timedelta(seconds=f1[i]) for i in range(f1.shape[0])])

        ds.close()

        # if 'foot1lon' dimension comes before 'foot1lat', swap the dimensions
        # some netcdf files store the footprint as (nstep x lon x lat).
        # We need (nstep x lat x lon).  Switch lat, lon around here
        if n2 < n1:
            b = numpy.empty((grid.shape[0], grid.shape[2], grid.shape[1]))
            nrows = grid.shape[0]
            for i in range(nrows):
                b[i] = grid[i].T

            grid = b

        # reduce grid to 1x1 degree (assumes dx = dy. would need to change if not)
        if dx == 0.5:
            c = grid[:, 0:-1:2, 0:-1:2] + grid[:, 0:-1:2, 1::2] + grid[:, 1::2, 0:-1:2] + grid[:, 1::2, 1::2]
        else:
            c = grid

        # extract inversion domain from the footprint domain
        # in case the footprint domain is different size than inversion domain
        if extract_domain:
            c = self.extract_grid(c, lats, lons)

        return c, dates, lats, lons

    #--------------------------------------------------------------
    def extract_grid(self, grid, lats, lons):
        """ Extract the inversion domain from the footprint domain

        Parameters
        ----------
        grid : array
            footprint grid [ntimesteps x nlatx x nlons]
        lats : array
            the latitudes in the footprint
        lons : array
            the longitudes in the footprint

        Returns
        -------
        grid : array
            inversion grid defined in config file

        Only works with 1 degree grids
        """

        # get the minimum latitude and longitude
        # use math.floor in case the latitudes are centered on a grid cell,
        # such that the value has .5 in it, e.g. -169.5
        latmin = math.floor(lats[0])
        lonmin = math.floor(lons[0])

        # determine the offset from the footprint domain where the inversion domain is located
        # if the footprint domain and inversion domain are the same, bottom and left are 0,
        # top and right are size of domain
        by = self.south - latmin	# bottom
        ty = self.north - latmin	# top
        lx = self.west - lonmin		# left
        rx = self.east - lonmin		# right

        mygrid = grid[:, by:ty, lx:rx]

        return mygrid

    #--------------------------------------------------------------
    def land_grid(self, grid):
        """ Extract land cells from footprint grid.

        Uses the landmask specified in the config file to extract the
        grid cells that are designated as land.

        Parameters
        ----------
        grid : array
            The full inversion grid.

        Results
        -------
        landgrid : array
            2d array numobs x numlandcells with footprint values.
        """

        return grid[:, self.landmask[0], self.landmask[1]]

    #--------------------------------------------------------------
    def load_file(self, filename, varname=None):
        """ Read in a data file, which can be either a
        numpy save file or a netcdf file.

        Parameters
        ----------
        filename : str
            name of file to load
        varname : str
            If filename is a netcdf file, varname must be set to the name
            of the variable in the netcdf file.

        Returns
        -------
        data : array
            data from the file
        """

        try:
            data = numpy.load(filename)
        except (ValueError, FileNotFoundError) as e:
            try:
                data = self._get_netcdf(filename, varname)
            except Exception as e:
                sys.exit("Cannot read file %s. %s" % (filename, e), file=sys.stderr)

        return data


    #--------------------------------------------------------------
    def _get_netcdf(self, filename, varname):
        """ Get grid values from a netcdf file.

        Use landmask to extract land cell values.
        Assumes that the netcdf data grid is compatible with landmask, e.g. 70x120 1 degree cells.

        Parameters
        ----------
        filename : str
            Name of netcdf file
        varname : str
            Name of variable in netcdf file to get.

        Returns
        -------
        data : array
            variable data from netcdf file
        """

        try:
            ds = netCDF4.Dataset(filename)
        except IOError as e:
            sys.exit("Error trying to read netcdf file %s. %s" % (filename, e))

        g = ds.variables[varname][:]
        data = numpy.array(g[:, self.landmask[0], self.landmask[1]])  # python 3 requires a cast to numpy array.

        ds.close()

        return data

    #--------------------------------------------------------------
    def makeTemporalCovariance(self, length=None):
        """ Compute temporal covariance matrix

        Algorithm from vineet's fortran 90 code,
        subroutine temporal_covariance in library_inverse.f90

        Parameters
        ----------
        length : int
            correlation length value, whose units are number of days

        Returns
        -------
        temp_cov : array
            array ntimesteps x ntimesteps of temporal covariance values

        Within day correlations will be 0, day to day
        correlations will be > 0.
        So e.g. 8 hour timestep, every 3rd value > 0,
        correlations will look like

        1 0 0 n 0 0 n 0 0
        0 1 0 0 n 0 0 n 0
        0 0 1 0 0 n 0 0 n
        n 0 0 1 0 0 n 0 0
        0 n 0 0 1 0 0 n 0
        0 0 n 0 0 1 0 0 n
        n 0 0 n 0 0 1 0 0
        0 n 0 0 n 0 0 1 0
        0 0 n 0 0 n 0 0 1

        where 0 < n < 1
        """

        withinday = int(self.steps_per_day)
        ntimesteps = self.ntimesteps
        ndays = self.ndays  # total number of days in inversion
        if length is None:
            temporal_cl = self.temporal_corr_length
        else:
            temporal_cl = length


        # if one or more timesteps per day, build array with 0 for within day correlation
        if withinday >= 1:

            # create an array size = ndays, fill out with day number
            t_dist = numpy.empty((ndays, ndays))
            for i in range(ndays):
                t_dist[i, i:] = numpy.arange(ndays-i)
                t_dist[i, 0:i] = numpy.arange(i, 0, -1)

            # apply correlation length
            if temporal_cl > 0:
                t_dist = numpy.exp(-t_dist/temporal_cl)
            else:
                t_dist = numpy.identity(ndays)

            # expand to ntimesteps x ntimesteps, with cells within same day = 0
            temp_cov = numpy.kron(t_dist, numpy.identity(withinday))

        else:

            # create an array size = ntimesteps, fill out with timestep number
            t_dist = numpy.empty((ntimesteps, ntimesteps))
            for i in range(ntimesteps):
                t_dist[i, i:] = numpy.arange(ntimesteps-i)
                t_dist[i, 0:i] = numpy.arange(i, 0, -1)

            # convert time step number to days
            days_per_step = 1.0 / withinday
            t_dist = t_dist*days_per_step

            # apply correlation length
            temp_cov = numpy.exp(-t_dist/temporal_cl)

        return temp_cov

    #--------------------------------------------------------------
    def get_sp_cov(self):
        """ Read in the spatial covariance file and return the data.

        The name of the file is specified as 'sp_cov_file' in the config file.

        The spatial covariance contains the distances between each
        of the land cells in the domain. It is created with the make_sc.py program.

        Returns
        -------
        heff : array
            spatial covariance data of shape nlandcells x nlandcells
        """

        spfile = "/".join([self.workdir, self.sp_cov_file])
        heff = numpy.load(spfile)

        return heff

    #--------------------------------------------------------------
    def convolve(self, data, hdir=None):
        """ Compute footprint H values * data

        Parameters
        ----------
        data : array
            Data to convolve
        hdir : str
            If set, this is directory name of H strip files.

        Returns
        -------
        hs : array
            convolved data H*data, shape is nobs x 1
        """

        hs = None
        timesteps = self.ntimesteps

        for i in range(timesteps):

            s = data[i]            # data at this timestep
            h = self.read_sparse_h(i, hdir=hdir)

            if hs is None:
                num_obs = h.shape[0]
                hs = numpy.zeros(num_obs)


#           hs = hs + numpy.ravel((h * s))  # the ravel changes nrows x 1 column to 1 row x ncolumns
            hs = hs + h * s

        return hs

    #--------------------------------------------------------------
    def make_ncdf(self, filename, name, shat):
        """ Convert a numpy array of shape #timesteps x #cells
        (#timesteps x (latxlon)) to a netcdf file.

        Parameters
        ----------
        filename - str
            netcdf output file name
        name : str
            name to give netcdf variable
        shat : array
            numpy data array

        """

        # convert shat dimensions #timesteps x #cells to
        # #timesteps x lats x lons for full domain

        nlat = self.nlatgrids
        nlon = self.nlongrids
        ntimesteps = self.ntimesteps

        print(nlat)
        print(nlon)
        print(ntimesteps)

        dates = []
        date = self.sdate
        while date < self.edate:
            dates.append(date)
            # dates.append(date + config["timestep"]/2)  # set dates at middle of time step interval
            date = date + self.timestep

        grid = numpy.zeros((ntimesteps, nlat, nlon))
        if self.debug:
            print("netcdf grid shape is ", grid.shape)
        for i in range(ntimesteps):
            a = shat[i]
            for num, val in enumerate(a):
                latidx = self.landmask[0, num]
                lonidx = self.landmask[1, num]
                grid[i, latidx, lonidx] = val

        ds = netCDF4.Dataset(filename, 'w', format='NETCDF4')

        nchar = 500
        ds.createDimension('single', 1)
        ds.createDimension('nchar', nchar)
        ds.createDimension('nlat', nlat)
        ds.createDimension('nlon', nlon)
        ds.createDimension('ndates', ntimesteps)

        #----------
        lats = ds.createVariable('lat', 'double', ('nlat'))
        lats.units = "degrees_north"
        lats.long_name = "latitude"
        lats.description = "latitude of center of cells"
        latmin = self.south
        latmax = self.north
        lats[:] = numpy.arange(latmin, latmax+self.lat_resolution/2, self.lat_resolution) 
        #----------
        latdelta = ds.createVariable('lat_delta', 'double', ('single'))
        latdelta.units = "degrees"
        latdelta.long_name = "size of cell latitude in degrees"
        latdelta[:] = self.lat_resolution

        #----------
        lons = ds.createVariable('lon', 'double', ('nlon'))
        lons.units = "degrees_east"
        lons.long_name = "longitude"
        lons.description = "longitude of center of cells"
        lonmin = self.west
        lonmax = self.east
        lons[:] = numpy.arange(lonmin, lonmax+self.lon_resolution/2, self.lon_resolution)
        #----------
        londelta = ds.createVariable('lon_delta', 'double', ('single'))
        londelta.units = "degrees"
        londelta.long_name = "size of cell longitude in degrees"
        londelta[:] = self.lon_resolution

        #----------
        foot1 = ds.createVariable(name, 'float', ('ndates', 'nlat', 'nlon'), fill_value=-1.e+34)
        foot1.units = "micromol m-2 s-1"
        foot1.long_name = name + " output"
        foot1[:] = grid

        #----------
        d = ds.createVariable('dates', 'float', 'ndates', fill_value=-1.e+34)
        d.units = "days since 2000-01-01 00:00:00 UTC"
        d.long_name = "dates"

        basedate = datetime.datetime(2000, 1, 1)
        date = []
        for dt in dates:
            x = dt - basedate
            diff = x.days + x.seconds/86400.0
            date.append(diff)

        d[:] = numpy.array(date)

        ds.close()
