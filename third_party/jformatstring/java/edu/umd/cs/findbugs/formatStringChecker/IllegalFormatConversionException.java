/*
 * Copyright 2003-2007 Sun Microsystems, Inc.  All Rights Reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Sun designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Sun in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Sun Microsystems, Inc., 4150 Network Circle, Santa Clara,
 * CA 95054 USA or visit www.sun.com if you need additional information or
 * have any questions.
 */
/* This file has been extensively modified from the original Sun implementation
 * to provide for compile time checking of Format Strings.
 * 
 * These modifications were performed by Bill Pugh, this code is free software.
 * 
 */
package edu.umd.cs.findbugs.formatStringChecker;

public class IllegalFormatConversionException extends FormatterException {

	private static final long serialVersionUID = 1L;

	final private String formatSpecifier;
	final private char conversion;
	final private String signature;
	int argIndex = -1;

	/**
	 * Constructs an instance of this class with the mismatched conversion and
	 * the corresponding argument class.
	 * 
	 * @param formatSpecifier
	 *            Inapplicable format specifier
	 * 
	 * @param signature
	 *            Signature of the mismatched argument
	 */
	public IllegalFormatConversionException(String formatSpecifier,
			char conversion, String signature) {
		if (signature == null)
			throw new NullPointerException();

		this.conversion = conversion;
		this.formatSpecifier = formatSpecifier;
		this.signature = signature;
	}

	public void setArgIndex(int argIndex) {
		if (argIndex == -1)
			throw new IllegalStateException("arg index already set");
		this.argIndex = argIndex;
	}

	public int getArgIndex() {
		return argIndex;
	}

	public String getFormatSpecifier() {
		return formatSpecifier;
	}

	public char getConversion() {
		return conversion;
	}

	/**
	 * Returns the class of the mismatched argument.
	 * 
	 * @return The class of the mismatched argument
	 */
	public String getArgumentSignature() {
		return signature;
	}

	// javadoc inherited from Throwable.java
	public String getMessage() {
		return String.format("%s can't format %s", formatSpecifier, signature);
	}

}
