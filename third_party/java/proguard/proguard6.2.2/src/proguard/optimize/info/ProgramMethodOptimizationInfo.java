/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
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
import proguard.util.ArrayUtil;

/**
 * This class stores some optimization information that can be attached to
 * a method.
 *
 * @author Eric Lafortune
 */
public class ProgramMethodOptimizationInfo
extends      MethodOptimizationInfo
{
    private static final Value[] EMPTY_PARAMETERS = new Value[0];


    private volatile boolean hasSideEffects           = false;
    private volatile boolean canBeMadePrivate         = true;
    private volatile boolean catchesExceptions        = false;
    private volatile boolean branchesBackward         = false;
    private volatile boolean invokesSuperMethods      = false;
    private volatile boolean invokesDynamically       = false;
    private volatile boolean accessesPrivateCode      = false;
    private volatile boolean accessesPackageCode      = false;
    private volatile boolean accessesProtectedCode    = false;
    private volatile boolean hasSynchronizedBlock     = false;
    private volatile boolean assignsFinalField        = false;
    private volatile boolean returnsWithNonEmptyStack = false;
    private volatile int     invocationCount          = 0;
    private volatile int     parameterSize            = 0;
    private volatile long    usedParameters           = 0L;
    private volatile long    escapedParameters        = 0L;
    private volatile long    escapingParameters       = 0L;
    private volatile long    modifiedParameters       = 0L;
    private volatile boolean modifiesAnything         = false;
    private volatile Value[] parameters;
    private volatile long    returnedParameters       = 0L;
    private volatile boolean returnsNewInstances      = false;
    private volatile boolean returnsExternalValues    = false;


    /**
     * Creates a new MethodOptimizationInfo for the given method.
     */
    public ProgramMethodOptimizationInfo(Clazz clazz, Method method)
    {
        // Set up an array of the right size for storing information about the
        // passed parameters (including 'this', for non-static methods).
        int parameterCount =
            ClassUtil.internalMethodParameterCount(method.getDescriptor(clazz),
                                                   method.getAccessFlags());

        parameters = parameterCount == 0 ?
            EMPTY_PARAMETERS :
            new Value[parameterCount];
    }


    public boolean isKept()
    {
        return false;
    }


    public void setSideEffects()
    {
        hasSideEffects = true;
    }


    public boolean hasSideEffects()
    {
        return !hasNoSideEffects && hasSideEffects;
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


    public void setHasSynchronizedBlock()
    {
        hasSynchronizedBlock = true;
    }


    public boolean hasSynchronizedBlock()
    {
        return hasSynchronizedBlock;
    }


    public void setAssignsFinalField()
    {
        assignsFinalField = true;
    }


    public boolean assignsFinalField()
    {
        return assignsFinalField;
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


    public synchronized void setParameterSize(int parameterSize)
    {
        this.parameterSize = parameterSize;
    }


    public int getParameterSize()
    {
        return parameterSize;
    }


    public synchronized void setParameterUsed(int variableIndex)
    {
        usedParameters = setBit(usedParameters, variableIndex);
    }


    public synchronized void updateUsedParameters(long usedParameters)
    {
        this.usedParameters |= usedParameters;
    }


    public boolean hasUnusedParameters()
    {
        return parameterSize < 64 ?
                  (usedParameters | -1L << parameterSize) != -1L :
                   usedParameters                         != -1L ;
    }


    public boolean isParameterUsed(int variableIndex)
    {
        return isBitSet(usedParameters, variableIndex);
    }


    public long getUsedParameters()
    {
        return usedParameters;
    }


    /**
     * Notifies this object that a parameter is inserted at the given
     * index.
     * @param parameterIndex the parameter index,
     *                       not taking into account the entry size,
     *                       but taking into account the 'this' parameter,
     *                       if any.
     */
    public synchronized void insertParameter(int parameterIndex)
    {
        // The used parameter bits are indexed with their variable indices
        // (which take into account the sizes of the entries).
        //usedParameters   = insertBit(usedParameters,     parameterIndex, 1L);
        //parameterSize++;

        escapedParameters  = insertBit(escapedParameters,  parameterIndex, 1L);
        escapingParameters = insertBit(escapingParameters, parameterIndex, 1L);
        modifiedParameters = insertBit(modifiedParameters, parameterIndex, 1L);
        returnedParameters = insertBit(returnedParameters, parameterIndex, 1L);
        parameters         = ArrayUtil.insert(parameters, parameters.length, parameterIndex, null);
    }


    /**
     * Notifies this object that the specified parameter is removed.
     * @param parameterIndex the parameter index,
     *                       not taking into account the entry size,
     *                       but taking into account the 'this' parameter,
     *                       if any.
     */
    public synchronized void removeParameter(int parameterIndex)
    {
        // The used parameter bits are indexed with their variable indices
        // (which take into account the sizes of the entries).
        //usedParameters   = removeBit(usedParameters,     parameterIndex, 1L);
        //parameterSize--;

        escapedParameters  = removeBit(escapedParameters,  parameterIndex, 1L);
        escapingParameters = removeBit(escapingParameters, parameterIndex, 1L);
        modifiedParameters = removeBit(modifiedParameters, parameterIndex, 1L);
        returnedParameters = removeBit(returnedParameters, parameterIndex, 1L);
        ArrayUtil.remove(parameters, parameters.length, parameterIndex);
    }


    public synchronized void setParameterEscaped(int parameterIndex)
    {
        escapedParameters = setBit(escapedParameters, parameterIndex);
    }


    public synchronized void updateEscapedParameters(long escapedParameters)
    {
        this.escapedParameters |= escapedParameters;
    }


    public boolean hasParameterEscaped(int parameterIndex)
    {
        return isBitSet(escapedParameters, parameterIndex);
    }


    public long getEscapedParameters()
    {
        return escapedParameters;
    }


    public synchronized void setParameterEscaping(int parameterIndex)
    {
        escapingParameters = setBit(escapingParameters, parameterIndex);
    }


    public synchronized void updateEscapingParameters(long escapingParameters)
    {
        this.escapingParameters |= escapingParameters;
    }


    public boolean isParameterEscaping(int parameterIndex)
    {
        return
            !hasNoEscapingParameters &&
            (isBitSet(escapingParameters, parameterIndex));
    }


    public long getEscapingParameters()
    {
        return hasNoEscapingParameters ? 0L : escapingParameters;
    }


    public synchronized void setParameterModified(int parameterIndex)
    {
        modifiedParameters = setBit(modifiedParameters, parameterIndex);
    }


    public synchronized void updateModifiedParameters(long modifiedParameters)
    {
        this.modifiedParameters |= modifiedParameters;
    }


    public boolean isParameterModified(int parameterIndex)
    {
        // TODO: Refine for static methods.
        return
            !hasNoSideEffects &&
            (!hasNoExternalSideEffects || parameterIndex == 0) &&
            (isBitSet((modifiesAnything ?
                  modifiedParameters | escapedParameters :
                  modifiedParameters), parameterIndex));
    }


    public long getModifiedParameters()
    {
        // TODO: Refine for static methods.
        return
            hasNoSideEffects         ? 0L :
            hasNoExternalSideEffects ? modifiedParameters & 1L :
                                       modifiedParameters;
    }


    public void setModifiesAnything()
    {
        modifiesAnything = true;
    }


    public boolean modifiesAnything()
    {
        return !hasNoExternalSideEffects && modifiesAnything;
    }


    public synchronized void generalizeParameterValue(int parameterIndex, Value parameter)
    {
        parameters[parameterIndex] = parameters[parameterIndex] != null ?
            parameters[parameterIndex].generalize(parameter) :
            parameter;
    }


    public Value getParameterValue(int parameterIndex)
    {
        return parameters != null ?
            parameters[parameterIndex] :
            null;
    }


    public synchronized void setParameterReturned(int parameterIndex)
    {
        returnedParameters = setBit(returnedParameters, parameterIndex);
    }


    public synchronized void updateReturnedParameters(long returnedParameters)
    {
        this.returnedParameters |= returnedParameters;
    }


    public boolean returnsParameter(int parameterIndex)
    {
        return isBitSet(returnedParameters, parameterIndex);
    }


    public long getReturnedParameters()
    {
        return returnedParameters;
    }


    public void setReturnsNewInstances()
    {
        returnsNewInstances = true;
    }


    public boolean returnsNewInstances()
    {
        return returnsNewInstances;
    }


    public void setReturnsExternalValues()
    {
        returnsExternalValues = true;
    }


    public boolean returnsExternalValues()
    {
        return
            !hasNoExternalReturnValues &&
            returnsExternalValues;
    }


    public synchronized void generalizeReturnValue(Value returnValue)
    {
        this.returnValue = this.returnValue != null ?
            this.returnValue.generalize(returnValue) :
            returnValue;
    }


    public synchronized void merge(MethodOptimizationInfo other)
    {
        this.catchesExceptions     |= other.catchesExceptions();
        this.branchesBackward      |= other.branchesBackward();
        this.invokesSuperMethods   |= other.invokesSuperMethods();
        this.invokesDynamically    |= other.invokesDynamically();
        this.accessesPrivateCode   |= other.accessesPrivateCode();
        this.accessesPackageCode   |= other.accessesPackageCode();
        this.accessesProtectedCode |= other.accessesProtectedCode();
        this.hasSynchronizedBlock  |= other.hasSynchronizedBlock();
        this.assignsFinalField     |= other.assignsFinalField();

        // Some of these should actually be recomputed, since these are
        // relative to the method:
        //     this.invokesSuperMethods
        //     this.accessesPrivateCode
        //     this.accessesPackageCode
        //     this.accessesProtectedCode
    }


    public static void setProgramMethodOptimizationInfo(Clazz clazz, Method method)
    {
        MethodLinker.lastMember(method).setVisitorInfo(new ProgramMethodOptimizationInfo(clazz, method));
    }


    public static ProgramMethodOptimizationInfo getProgramMethodOptimizationInfo(Method method)
    {
        return (ProgramMethodOptimizationInfo)MethodLinker.lastMember(method).getVisitorInfo();
    }


    // Small utility methods.

    /**
     * Returns the given value with the specified bit set.
     */
    private long setBit(long bits, int index)
    {
        return index < 64 ?
            bits | (1L << index) :
            bits;
    }


    /**
     * Returns whether the specified bit is set in the given value
     * (or if the index exceeds the size of the long).
     */
    private boolean isBitSet(long bits, int index)
    {
        return index >= 64 || (bits & (1L << index)) != 0;
    }


    /**
     * Returns the given value with a given bit inserted at the given index.
     */
    private long insertBit(long value, int bitIndex, long bitValue)
    {
        long higherMask = -1L << bitIndex;
        long lowerMask  = ~higherMask;

        return ((value & higherMask) << 1) |
               ( value & lowerMask       ) |
               (bitValue << bitIndex);
    }


    /**
     * Returns the given value with a bit removed at the given index.
     * The given given bit value is shifted in as the new most significant bit.
     */
    private long removeBit(long value, int bitIndex, long highBitValue)
    {
        long higherMask = -1L << bitIndex;
        long lowerMask  = ~higherMask;

        return ((value & (higherMask<<1)) >>> 1) |
               ( value & lowerMask             ) |
               (highBitValue << 63);
    }
}
