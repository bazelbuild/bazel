/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.optimize.info;

import proguard.classfile.*;
import proguard.classfile.util.*;
import proguard.evaluation.value.Value;

/**
 * This class stores some optimization information that can be attached to
 * a method.
 *
 * @author Eric Lafortune
 */
public class MethodOptimizationInfo
{
    private boolean hasNoSideEffects         = false;
    private boolean hasSideEffects           = false;
    private boolean canBeMadePrivate         = true;
    private boolean catchesExceptions        = false;
    private boolean branchesBackward         = false;
    private boolean invokesSuperMethods      = false;
    private boolean invokesDynamically       = false;
    private boolean accessesPrivateCode      = false;
    private boolean accessesPackageCode      = false;
    private boolean accessesProtectedCode    = false;
    private boolean returnsWithNonEmptyStack = false;
    private int     invocationCount          = 0;
    private int     parameterSize            = 0;
    private long    usedParameters           = 0L;
    private Value[] parameters;
    private Value   returnValue;


    /**
     * Creates a new MethodOptimizationInfo for the given method.
     */
    public MethodOptimizationInfo(Clazz clazz, Method method)
    {
        // Set up an array of the right size for storing information about the
        // passed parameters.
        int parameterCount =
            ClassUtil.internalMethodParameterCount(method.getDescriptor(clazz));

        if ((method.getAccessFlags() & ClassConstants.ACC_STATIC) == 0)
        {
            parameterCount++;
        }

        if (parameterCount > 0)
        {
            parameters = new Value[parameterCount];
        }
    }


    public void setNoSideEffects()
    {
        hasNoSideEffects = true;
    }


    public boolean hasNoSideEffects()
    {
        return hasNoSideEffects;
    }


    public void setSideEffects()
    {
        hasSideEffects = true;
    }


    public boolean hasSideEffects()
    {
        return hasSideEffects;
    }


    public void setCanNotBeMadePrivate()
    {
        canBeMadePrivate = false;
    }


    public boolean canBeMadePrivate()
    {
        return canBeMadePrivate;
    }


    public void setCatchesExceptions()
    {
        catchesExceptions = true;
    }


    public boolean catchesExceptions()
    {
        return catchesExceptions;
    }


    public void setBranchesBackward()
    {
        branchesBackward = true;
    }


    public boolean branchesBackward()
    {
        return branchesBackward;
    }


    public void setInvokesSuperMethods()
    {
        invokesSuperMethods = true;
    }


    public boolean invokesSuperMethods()
    {
        return invokesSuperMethods;
    }


    public void setInvokesDynamically()
    {
        invokesDynamically = true;
    }


    public boolean invokesDynamically()
    {
        return invokesDynamically;
    }


    public void setAccessesPrivateCode()
    {
        accessesPrivateCode = true;
    }


    public boolean accessesPrivateCode()
    {
        return accessesPrivateCode;
    }


    public void setAccessesPackageCode()
    {
        accessesPackageCode = true;
    }


    public boolean accessesPackageCode()
    {
        return accessesPackageCode;
    }


    public void setAccessesProtectedCode()
    {
        accessesProtectedCode = true;
    }


    public boolean accessesProtectedCode()
    {
        return accessesProtectedCode;
    }


    public void setReturnsWithNonEmptyStack()
    {
        returnsWithNonEmptyStack = true;
    }


    public boolean returnsWithNonEmptyStack()
    {
        return returnsWithNonEmptyStack;
    }


    public void incrementInvocationCount()
    {
        invocationCount++;
    }


    public int getInvocationCount()
    {
        return invocationCount;
    }


    public void setParameterSize(int parameterSize)
    {
        this.parameterSize = parameterSize;
    }


    public int getParameterSize()
    {
        return parameterSize;
    }


    public void setParameterUsed(int parameterIndex)
    {
        usedParameters |= 1L << parameterIndex;
    }


    public void setUsedParameters(long usedParameters)
    {
        this.usedParameters = usedParameters;
    }


    public boolean isParameterUsed(int parameterIndex)
    {
        return parameterIndex >= 64 || (usedParameters & (1L << parameterIndex)) != 0;
    }


    public long getUsedParameters()
    {
        return usedParameters;
    }


    public void generalizeParameter(int parameterIndex, Value parameter)
    {
        parameters[parameterIndex] = parameters[parameterIndex] != null ?
            parameters[parameterIndex].generalize(parameter) :
            parameter;
    }


    public Value getParameter(int parameterIndex)
    {
        return parameters != null ?
            parameters[parameterIndex] :
            null;
    }


    public void generalizeReturnValue(Value returnValue)
    {
        this.returnValue = this.returnValue != null ?
            this.returnValue.generalize(returnValue) :
            returnValue;
    }


    public Value getReturnValue()
    {
        return returnValue;
    }


    // For setting enum return values.
    public void setReturnValue(Value returnValue)
    {
        this.returnValue = returnValue;
    }


    public void merge(MethodOptimizationInfo other)
    {
        if (other != null)
        {
            this.hasNoSideEffects      &= other.hasNoSideEffects;
            this.hasSideEffects        |= other.hasSideEffects;
            //this.canBeMadePrivate    &= other.canBeMadePrivate;
            this.catchesExceptions     |= other.catchesExceptions;
            this.branchesBackward      |= other.branchesBackward;
            this.invokesSuperMethods   |= other.invokesSuperMethods;
            this.invokesDynamically    |= other.invokesDynamically;
            this.accessesPrivateCode   |= other.accessesPrivateCode;
            this.accessesPackageCode   |= other.accessesPackageCode;
            this.accessesProtectedCode |= other.accessesProtectedCode;
        }
        else
        {
            this.hasNoSideEffects      = false;
            this.hasSideEffects        = true;
            //this.canBeMadePrivate    = false;
            this.catchesExceptions     = true;
            this.branchesBackward      = true;
            this.invokesSuperMethods   = true;
            this.accessesPrivateCode   = true;
            this.accessesPackageCode   = true;
            this.accessesProtectedCode = true;
        }
    }


    public static void setMethodOptimizationInfo(Clazz clazz, Method method)
    {
        MethodLinker.lastMember(method).setVisitorInfo(new MethodOptimizationInfo(clazz, method));
    }


    public static MethodOptimizationInfo getMethodOptimizationInfo(Method method)
    {
        Object visitorInfo = MethodLinker.lastMember(method).getVisitorInfo();

        return visitorInfo instanceof MethodOptimizationInfo ?
            (MethodOptimizationInfo)visitorInfo :
            null;
    }
}
