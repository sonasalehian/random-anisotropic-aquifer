# trace generated using paraview version 5.10.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 10

#### import the simple module from the paraview
from paraview.simple import *

random_types = ["random_scaling_and_rotation", "random_scaling", "random_rotation"]
statistics = ["mean", "std"]

for random_type in random_types:
    for statistic in statistics:

        #### disable automatic camera reset on 'Show'
        paraview.simple._DisableFirstRenderCameraReset()

        # create a new 'ADIOS2VTXReader'
        meanbp = ADIOS2VTXReader(registrationName=f'{statistic}.bp', FileName=f'output/{random_type}/{statistic}.bp')*1000

        # get animation scene
        animationScene1 = GetAnimationScene()

        # update animation scene based on data timesteps
        animationScene1.UpdateAnimationUsingDataTimeSteps()

        # get active view
        renderView1 = GetActiveViewOrCreate('RenderView')

        # show data in view
        meanbpDisplay = Show(meanbp, renderView1, 'UnstructuredGridRepresentation')

        # trace defaults for the display properties.
        meanbpDisplay.Representation = 'Surface'
        meanbpDisplay.ColorArrayName = [None, '']
        meanbpDisplay.SelectTCoordArray = 'None'
        meanbpDisplay.SelectNormalArray = 'None'
        meanbpDisplay.SelectTangentArray = 'None'
        meanbpDisplay.OSPRayScaleArray = 'f'
        meanbpDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
        meanbpDisplay.SelectOrientationVectors = 'None'
        meanbpDisplay.ScaleFactor = 350.00000000000006
        meanbpDisplay.SelectScaleArray = 'None'
        meanbpDisplay.GlyphType = 'Arrow'
        meanbpDisplay.GlyphTableIndexArray = 'None'
        meanbpDisplay.GaussianRadius = 17.500000000000004
        meanbpDisplay.SetScaleArray = ['POINTS', 'f']
        meanbpDisplay.ScaleTransferFunction = 'PiecewiseFunction'
        meanbpDisplay.OpacityArray = ['POINTS', 'f']
        meanbpDisplay.OpacityTransferFunction = 'PiecewiseFunction'
        meanbpDisplay.DataAxesGrid = 'GridAxesRepresentation'
        meanbpDisplay.PolarAxes = 'PolarAxesRepresentation'
        meanbpDisplay.ScalarOpacityUnitDistance = 155.43522262224357
        meanbpDisplay.OpacityArrayName = ['POINTS', 'f']

        # init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
        meanbpDisplay.OSPRayScaleFunction.Points = [2.1729651585822833e-11, 0.0, 0.5, 0.0, 5.3463965968432026e-05, 1.0, 0.5, 0.0]

        # init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
        meanbpDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

        # init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
        meanbpDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

        # reset view to fit data
        renderView1.ResetCamera(False)

        # get the material library
        materialLibrary1 = GetMaterialLibrary()

        # update the view to ensure updated data information
        renderView1.Update()

        # set scalar coloring
        ColorBy(meanbpDisplay, ('POINTS', 'f'))

        # rescale color and/or opacity maps used to include current data range
        meanbpDisplay.RescaleTransferFunctionToDataRange(True, False)

        # show color bar/color legend
        meanbpDisplay.SetScalarBarVisibility(renderView1, True)

        # get color transfer function/color map for 'f'
        fLUT = GetColorTransferFunction('f')

        # get opacity transfer function/opacity map for 'f'
        fPWF = GetOpacityTransferFunction('f')

        # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
        fLUT.ApplyPreset('Viridis (matplotlib)', True)

        if statistic == "mean":
            # invert the transfer function
            fLUT.InvertTransferFunction()

            # Rescale transfer function
            fLUT.RescaleTransferFunction(-0.00222, 0.0)

            # Rescale transfer function
            fPWF.RescaleTransferFunction(-0.00222, 0.0)

        elif statistic == "std":

            # Rescale transfer function
            fLUT.RescaleTransferFunction(0.0, 4.2e-04)

            # Rescale transfer function
            fPWF.RescaleTransferFunction(0.0, 4.2e-04)

        # reset view to fit data
        renderView1.ResetCamera(False)

        # Properties modified on animationScene1
        animationScene1.AnimationTime = 345600.0

        # get the time-keeper
        timeKeeper1 = GetTimeKeeper()

        # # Apply a preset using its name. Note this may not work as expected when presets have duplicate names.
        # fLUT.ApplyPreset('Viridis (matplotlib)', True)

        # get layout
        layout1 = GetLayout()

        # layout/tab size in pixels
        layout1.SetSize(970, 793)

        # current camera placement for renderView1
        renderView1.CameraPosition = [1000.0, 1750.0, -7767.926354575162]
        renderView1.CameraFocalPoint = [1000.0, 1750.0, 20.0]
        renderView1.CameraViewUp = [0.3420201433256687, 0.9396926207859084, 0.0]
        renderView1.CameraParallelScale = 2015.6636624198989

        # save screenshot
        SaveScreenshot(f'output/{random_type}/{statistic}.png', renderView1, ImageResolution=[3880, 3172])

        #================================================================
        # addendum: following script captures some of the application
        # state to faithfully reproduce the visualization during playback
        #================================================================

        #--------------------------------
        # saving layout sizes for layouts

        # layout/tab size in pixels
        layout1.SetSize(970, 793)

        #-----------------------------------
        # saving camera placements for views

        # current camera placement for renderView1
        renderView1.CameraPosition = [1000.0, 1750.0, -7767.926354575162]
        renderView1.CameraFocalPoint = [1000.0, 1750.0, 20.0]
        renderView1.CameraViewUp = [0.3420201433256687, 0.9396926207859084, 0.0]
        renderView1.CameraParallelScale = 2015.6636624198989

        #--------------------------------------------
        # uncomment the following to render all views
        # RenderAllViews()
        # alternatively, if you want to write images, you can use SaveScreenshot(...).