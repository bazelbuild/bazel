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
package proguard.gui;

import proguard.MemberSpecification;
import proguard.classfile.*;
import proguard.classfile.util.ClassUtil;
import proguard.util.ListUtil;

import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.awt.event.*;

/**
 * This <code>JDialog</code> allows the user to enter a String.
 *
 * @author Eric Lafortune
 */
final class MemberSpecificationDialog extends JDialog
{
    /**
     * Return value if the dialog is canceled (with the Cancel button or by
     * closing the dialog window).
     */
    public static final int CANCEL_OPTION = 1;

    /**
     * Return value if the dialog is approved (with the Ok button).
     */
    public static final int APPROVE_OPTION = 0;


    private final boolean isField;

    private final JRadioButton[] publicRadioButtons;
    private final JRadioButton[] privateRadioButtons;
    private final JRadioButton[] protectedRadioButtons;
    private final JRadioButton[] staticRadioButtons;
    private final JRadioButton[] finalRadioButtons;
    private final JRadioButton[] syntheticRadioButtons;

    private JRadioButton[] volatileRadioButtons;
    private JRadioButton[] transientRadioButtons;

    private JRadioButton[] synchronizedRadioButtons;
    private JRadioButton[] nativeRadioButtons;
    private JRadioButton[] abstractRadioButtons;
    private JRadioButton[] strictRadioButtons;
    private JRadioButton[] bridgeRadioButtons;
    private JRadioButton[] varargsRadioButtons;

    private final JTextField annotationTypeTextField = new JTextField(20);
    private final JTextField nameTextField           = new JTextField(20);
    private final JTextField typeTextField           = new JTextField(20);
    private final JTextField argumentTypesTextField  = new JTextField(20);

    private int returnValue;


    public MemberSpecificationDialog(JDialog owner, boolean isField)
    {
        super(owner, msg(isField ? "specifyFields" : "specifyMethods"), true);
        setResizable(true);

        // Create some constraints that can be reused.
        GridBagConstraints constraints = new GridBagConstraints();
        constraints.anchor = GridBagConstraints.WEST;
        constraints.insets = new Insets(1, 2, 1, 2);

        GridBagConstraints constraintsStretch = new GridBagConstraints();
        constraintsStretch.fill    = GridBagConstraints.HORIZONTAL;
        constraintsStretch.weightx = 1.0;
        constraintsStretch.anchor  = GridBagConstraints.WEST;
        constraintsStretch.insets  = constraints.insets;

        GridBagConstraints constraintsLast = new GridBagConstraints();
        constraintsLast.gridwidth = GridBagConstraints.REMAINDER;
        constraintsLast.anchor    = GridBagConstraints.WEST;
        constraintsLast.insets    = constraints.insets;

        GridBagConstraints constraintsLastStretch = new GridBagConstraints();
        constraintsLastStretch.gridwidth = GridBagConstraints.REMAINDER;
        constraintsLastStretch.fill      = GridBagConstraints.HORIZONTAL;
        constraintsLastStretch.weightx   = 1.0;
        constraintsLastStretch.anchor    = GridBagConstraints.WEST;
        constraintsLastStretch.insets    = constraints.insets;

        GridBagConstraints panelConstraints = new GridBagConstraints();
        panelConstraints.gridwidth = GridBagConstraints.REMAINDER;
        panelConstraints.fill      = GridBagConstraints.HORIZONTAL;
        panelConstraints.weightx   = 1.0;
        panelConstraints.weighty   = 0.0;
        panelConstraints.anchor    = GridBagConstraints.NORTHWEST;
        panelConstraints.insets    = constraints.insets;

        GridBagConstraints stretchPanelConstraints = new GridBagConstraints();
        stretchPanelConstraints.gridwidth = GridBagConstraints.REMAINDER;
        stretchPanelConstraints.fill      = GridBagConstraints.BOTH;
        stretchPanelConstraints.weightx   = 1.0;
        stretchPanelConstraints.weighty   = 1.0;
        stretchPanelConstraints.anchor    = GridBagConstraints.NORTHWEST;
        stretchPanelConstraints.insets    = constraints.insets;

        GridBagConstraints labelConstraints = new GridBagConstraints();
        labelConstraints.anchor = GridBagConstraints.CENTER;
        labelConstraints.insets = new Insets(2, 10, 2, 10);

        GridBagConstraints lastLabelConstraints = new GridBagConstraints();
        lastLabelConstraints.gridwidth = GridBagConstraints.REMAINDER;
        lastLabelConstraints.anchor    = GridBagConstraints.CENTER;
        lastLabelConstraints.insets    = labelConstraints.insets;

        GridBagConstraints advancedButtonConstraints = new GridBagConstraints();
        advancedButtonConstraints.weightx = 1.0;
        advancedButtonConstraints.weighty = 1.0;
        advancedButtonConstraints.anchor  = GridBagConstraints.SOUTHWEST;
        advancedButtonConstraints.insets  = new Insets(4, 4, 8, 4);

        GridBagConstraints okButtonConstraints = new GridBagConstraints();
        okButtonConstraints.weightx = 1.0;
        okButtonConstraints.weighty = 1.0;
        okButtonConstraints.anchor  = GridBagConstraints.SOUTHEAST;
        okButtonConstraints.insets  = advancedButtonConstraints.insets;

        GridBagConstraints cancelButtonConstraints = new GridBagConstraints();
        cancelButtonConstraints.gridwidth = GridBagConstraints.REMAINDER;
        cancelButtonConstraints.weighty   = 1.0;
        cancelButtonConstraints.anchor    = GridBagConstraints.SOUTHEAST;
        cancelButtonConstraints.insets    = okButtonConstraints.insets;

        GridBagLayout layout = new GridBagLayout();

        Border etchedBorder = BorderFactory.createEtchedBorder(EtchedBorder.RAISED);

        this.isField = isField;

        // Create the access panel.
        JPanel accessPanel = new JPanel(layout);
        accessPanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                               msg("access")));

        accessPanel.add(Box.createGlue(),                                labelConstraints);
        accessPanel.add(tip(new JLabel(msg("required")), "requiredTip"), labelConstraints);
        accessPanel.add(tip(new JLabel(msg("not")),      "notTip"),      labelConstraints);
        accessPanel.add(tip(new JLabel(msg("dontCare")), "dontCareTip"), labelConstraints);
        accessPanel.add(Box.createGlue(),                                constraintsLastStretch);

        publicRadioButtons    = addRadioButtonTriplet("Public",    accessPanel);
        privateRadioButtons   = addRadioButtonTriplet("Private",   accessPanel);
        protectedRadioButtons = addRadioButtonTriplet("Protected", accessPanel);
        staticRadioButtons    = addRadioButtonTriplet("Static",    accessPanel);
        finalRadioButtons     = addRadioButtonTriplet("Final",     accessPanel);
        syntheticRadioButtons = addRadioButtonTriplet("Synthetic", accessPanel);

        if (isField)
        {
            volatileRadioButtons  = addRadioButtonTriplet("Volatile",  accessPanel);
            transientRadioButtons = addRadioButtonTriplet("Transient", accessPanel);
        }
        else
        {
            synchronizedRadioButtons = addRadioButtonTriplet("Synchronized", accessPanel);
            nativeRadioButtons       = addRadioButtonTriplet("Native",       accessPanel);
            abstractRadioButtons     = addRadioButtonTriplet("Abstract",     accessPanel);
            strictRadioButtons       = addRadioButtonTriplet("Strict",       accessPanel);
            bridgeRadioButtons       = addRadioButtonTriplet("Bridge",       accessPanel);
            varargsRadioButtons      = addRadioButtonTriplet("Varargs",      accessPanel);
        }

        // Create the type panel.
        JPanel typePanel = new JPanel(layout);
        typePanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                             msg(isField ? "fieldType" :
                                                                           "returnType")));

        typePanel.add(tip(typeTextField, "typeTip"), constraintsLastStretch);

        // Create the annotation type panel.
        final JPanel annotationTypePanel = new JPanel(layout);
        annotationTypePanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                                       msg("annotation")));

        annotationTypePanel.add(tip(annotationTypeTextField, "classNameTip"), constraintsLastStretch);

        // Create the name panel.
        JPanel namePanel = new JPanel(layout);
        namePanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                             msg("name")));

        namePanel.add(tip(nameTextField, isField ? "fieldNameTip" :
                                                   "methodNameTip"), constraintsLastStretch);

        // Create the arguments panel.
        JPanel argumentsPanel = new JPanel(layout);
        argumentsPanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                                  msg("argumentTypes")));

        argumentsPanel.add(tip(argumentTypesTextField, "argumentTypes2Tip"), constraintsLastStretch);

        // Create the Advanced button.
        final JButton advancedButton = new JButton(msg("basic"));
        advancedButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                boolean visible = !annotationTypePanel.isVisible();

                annotationTypePanel.setVisible(visible);

                advancedButton.setText(msg(visible ? "basic" : "advanced"));

                pack();
            }
        });
        advancedButton.doClick();

        // Create the Ok button.
        JButton okButton = new JButton(msg("ok"));
        okButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                returnValue = APPROVE_OPTION;
                hide();
            }
        });

        // Create the Cancel button.
        JButton cancelButton = new JButton(msg("cancel"));
        cancelButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                hide();
            }
        });

        // Add all panels to the main panel.
        JPanel mainPanel = new JPanel(layout);
        mainPanel.add(tip(accessPanel,         "accessTip"),       panelConstraints);
        mainPanel.add(tip(annotationTypePanel, "annotationTip"),   panelConstraints);
        mainPanel.add(tip(typePanel, isField ? "fieldTypeTip" :
                                               "returnTypeTip"),   panelConstraints);
        mainPanel.add(tip(namePanel,           "nameTip"),         panelConstraints);

        if (!isField)
        {
            mainPanel.add(tip(argumentsPanel, "argumentTypesTip"), panelConstraints);
        }

        mainPanel.add(tip(advancedButton, "advancedTip"), advancedButtonConstraints);
        mainPanel.add(okButton,                           okButtonConstraints);
        mainPanel.add(cancelButton,                       cancelButtonConstraints);

        getContentPane().add(new JScrollPane(mainPanel));
    }


    /**
     * Adds a JLabel and three JRadioButton instances in a ButtonGroup to the
     * given panel with a GridBagLayout, and returns the buttons in an array.
     */
    private JRadioButton[] addRadioButtonTriplet(String labelText,
                                                 JPanel panel)
    {
        GridBagConstraints labelConstraints = new GridBagConstraints();
        labelConstraints.anchor = GridBagConstraints.WEST;
        labelConstraints.insets = new Insets(2, 10, 2, 10);

        GridBagConstraints buttonConstraints = new GridBagConstraints();
        buttonConstraints.insets = labelConstraints.insets;

        GridBagConstraints lastGlueConstraints = new GridBagConstraints();
        lastGlueConstraints.gridwidth = GridBagConstraints.REMAINDER;
        lastGlueConstraints.weightx   = 1.0;

        // Create the radio buttons.
        JRadioButton radioButton0 = new JRadioButton();
        JRadioButton radioButton1 = new JRadioButton();
        JRadioButton radioButton2 = new JRadioButton();

        // Put them in a button group.
        ButtonGroup buttonGroup = new ButtonGroup();
        buttonGroup.add(radioButton0);
        buttonGroup.add(radioButton1);
        buttonGroup.add(radioButton2);

        // Add the label and the buttons to the panel.
        panel.add(new JLabel(labelText), labelConstraints);
        panel.add(radioButton0,          buttonConstraints);
        panel.add(radioButton1,          buttonConstraints);
        panel.add(radioButton2,          buttonConstraints);
        panel.add(Box.createGlue(),      lastGlueConstraints);

        return new JRadioButton[]
        {
             radioButton0,
             radioButton1,
             radioButton2
        };
    }


    /**
     * Sets the MemberSpecification to be represented in this dialog.
     */
    public void setMemberSpecification(MemberSpecification memberSpecification)
    {
        String annotationType = memberSpecification.annotationType;
        String name           = memberSpecification.name;
        String descriptor     = memberSpecification.descriptor;

        // Set the class name text fields.
        annotationTypeTextField.setText(annotationType == null ? "" : ClassUtil.externalType(annotationType));

        // Set the access radio buttons.
        setMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_PUBLIC,       publicRadioButtons);
        setMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_PRIVATE,      privateRadioButtons);
        setMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_PROTECTED,    protectedRadioButtons);
        setMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_STATIC,       staticRadioButtons);
        setMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_FINAL,        finalRadioButtons);
        setMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_SYNTHETIC,    syntheticRadioButtons);
        setMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_VOLATILE,     volatileRadioButtons);
        setMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_TRANSIENT,    transientRadioButtons);
        setMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_SYNCHRONIZED, synchronizedRadioButtons);
        setMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_NATIVE,       nativeRadioButtons);
        setMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_ABSTRACT,     abstractRadioButtons);
        setMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_STRICT,       strictRadioButtons);
        setMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_BRIDGE,       bridgeRadioButtons);
        setMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_VARARGS,      varargsRadioButtons);

        // Set the class name text fields.
        nameTextField.setText(name == null ? "*" : name);

        if (isField)
        {
            typeTextField         .setText(descriptor == null ? "***" : ClassUtil.externalType(descriptor));
        }
        else
        {
            typeTextField         .setText(descriptor == null ? "***" : ClassUtil.externalMethodReturnType(descriptor));
            argumentTypesTextField.setText(descriptor == null ? "..." : ClassUtil.externalMethodArguments(descriptor));
        }
    }


    /**
     * Returns the MemberSpecification currently represented in this dialog.
     */
    public MemberSpecification getMemberSpecification()
    {
        String annotationType = annotationTypeTextField.getText();
        String name           = nameTextField.getText();
        String type           = typeTextField.getText();
        String arguments      = argumentTypesTextField.getText();

        // Convert all class member specifications into the internal format.
        annotationType =
            annotationType.equals("") ||
            annotationType.equals("***") ? null : ClassUtil.internalType(annotationType);

        if (name.equals("") ||
            name.equals("*"))
        {
            name = null;
        }

        if (isField)
        {
            type =
                type.equals("") ||
                type.equals("***") ? null : ClassUtil.internalType(type);
        }
        else
        {
            if (type.equals(""))
            {
                type = JavaConstants.TYPE_VOID;
            }

            type =
                type     .equals("***") &&
                arguments.equals("...") ? null :
                    ClassUtil.internalMethodDescriptor(type, ListUtil.commaSeparatedList(arguments));
        }

        MemberSpecification memberSpecification =
            new MemberSpecification(0, 0, annotationType, name, type);

        // Also get the access radio button settings.
        getMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_PUBLIC,       publicRadioButtons);
        getMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_PRIVATE,      privateRadioButtons);
        getMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_PROTECTED,    protectedRadioButtons);
        getMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_STATIC,       staticRadioButtons);
        getMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_FINAL,        finalRadioButtons);
        getMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_SYNTHETIC,    syntheticRadioButtons);
        getMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_VOLATILE,     volatileRadioButtons);
        getMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_TRANSIENT,    transientRadioButtons);
        getMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_SYNCHRONIZED, synchronizedRadioButtons);
        getMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_NATIVE,       nativeRadioButtons);
        getMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_ABSTRACT,     abstractRadioButtons);
        getMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_STRICT,       strictRadioButtons);
        getMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_BRIDGE,       bridgeRadioButtons);
        getMemberSpecificationRadioButtons(memberSpecification, ClassConstants.ACC_VARARGS,      varargsRadioButtons);

        return memberSpecification;
    }


    /**
     * Shows this dialog. This method only returns when the dialog is closed.
     *
     * @return <code>CANCEL_OPTION</code> or <code>APPROVE_OPTION</code>,
     *         depending on the choice of the user.
     */
    public int showDialog()
    {
        returnValue = CANCEL_OPTION;

        // Open the dialog in the right place, then wait for it to be closed,
        // one way or another.
        pack();
        setLocationRelativeTo(getOwner());
        show();

        return returnValue;
    }


    /**
     * Sets the appropriate radio button of a given triplet, based on the access
     * flags of the given keep option.
     */
    private void setMemberSpecificationRadioButtons(MemberSpecification memberSpecification,
                                                    int                 flag,
                                                    JRadioButton[]      radioButtons)
    {
        if (radioButtons != null)
        {
            int index = (memberSpecification.requiredSetAccessFlags   & flag) != 0 ? 0 :
                        (memberSpecification.requiredUnsetAccessFlags & flag) != 0 ? 1 :
                                                                                       2;
            radioButtons[index].setSelected(true);
        }
    }


    /**
     * Updates the access flag of the given keep option, based on the given radio
     * button triplet.
     */
    private void getMemberSpecificationRadioButtons(MemberSpecification memberSpecification,
                                                    int                 flag,
                                                    JRadioButton[]      radioButtons)
    {
        if (radioButtons != null)
        {
            if      (radioButtons[0].isSelected())
            {
                memberSpecification.requiredSetAccessFlags   |= flag;
            }
            else if (radioButtons[1].isSelected())
            {
                memberSpecification.requiredUnsetAccessFlags |= flag;
            }
        }
    }


    /**
     * Attaches the tool tip from the GUI resources that corresponds to the
     * given key, to the given component.
     */
    private static JComponent tip(JComponent component, String messageKey)
    {
        component.setToolTipText(msg(messageKey));

        return component;
    }


    /**
     * Returns the message from the GUI resources that corresponds to the given
     * key.
     */
    private static String msg(String messageKey)
    {
         return GUIResources.getMessage(messageKey);
    }
}
