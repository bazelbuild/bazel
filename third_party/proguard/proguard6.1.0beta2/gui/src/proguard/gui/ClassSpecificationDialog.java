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

import proguard.*;
import proguard.classfile.ClassConstants;
import proguard.classfile.util.ClassUtil;

import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.awt.event.*;
import java.util.List;

/**
 * This <code>JDialog</code> enables the user to specify a class specification.
 *
 * @author Eric Lafortune
 */
final class ClassSpecificationDialog extends JDialog
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


    private final JTextArea commentsTextArea = new JTextArea(4, 20);

    private final JRadioButton keepClassesAndMembersRadioButton  = new JRadioButton(msg("keep"));
    private final JRadioButton keepClassMembersRadioButton       = new JRadioButton(msg("keepClassMembers"));
    private final JRadioButton keepClassesWithMembersRadioButton = new JRadioButton(msg("keepClassesWithMembers"));

    private final JCheckBox keepDescriptorClassesCheckBox = new JCheckBox(msg("keepDescriptorClasses"));
    private final JCheckBox keepCodeCheckBox              = new JCheckBox(msg("keepCode"));

    private final JCheckBox allowShrinkingCheckBox    = new JCheckBox(msg("allowShrinking"));
    private final JCheckBox allowOptimizationCheckBox = new JCheckBox(msg("allowOptimization"));
    private final JCheckBox allowObfuscationCheckBox  = new JCheckBox(msg("allowObfuscation"));

    private final JTextField               conditionCommentsField = new JTextField(20);
    private final ClassSpecificationDialog conditionDialog;

    private final JRadioButton[] publicRadioButtons;
    private final JRadioButton[] finalRadioButtons;
    private final JRadioButton[] abstractRadioButtons;
    private final JRadioButton[] interfaceRadioButtons;
    private final JRadioButton[] annotationRadioButtons;
    private final JRadioButton[] enumRadioButtons;
    private final JRadioButton[] syntheticRadioButtons;

    private final JTextField annotationTypeTextField        = new JTextField(20);
    private final JTextField classNameTextField             = new JTextField(20);
    private final JTextField extendsAnnotationTypeTextField = new JTextField(20);
    private final JTextField extendsClassNameTextField      = new JTextField(20);

    private final MemberSpecificationsPanel memberSpecificationsPanel;

    private int returnValue;


    public ClassSpecificationDialog(final JFrame owner,
                                    boolean      includeKeepSettings,
                                    boolean      includeFieldButton)
    {
        super(owner, msg("specifyClasses"), true);

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
        cancelButtonConstraints.insets    = advancedButtonConstraints.insets;

        GridBagLayout layout = new GridBagLayout();

        Border etchedBorder = BorderFactory.createEtchedBorder(EtchedBorder.RAISED);

        // Create the comments panel.
        JPanel commentsPanel = new JPanel(layout);
        commentsPanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                                 msg("comments")));

        JScrollPane commentsScrollPane = new JScrollPane(commentsTextArea);
        commentsScrollPane.setBorder(classNameTextField.getBorder());

        commentsPanel.add(tip(commentsScrollPane, "commentsTip"), constraintsLastStretch);

        // Create the keep option panel.
        ButtonGroup keepButtonGroup = new ButtonGroup();
        keepButtonGroup.add(keepClassesAndMembersRadioButton);
        keepButtonGroup.add(keepClassMembersRadioButton);
        keepButtonGroup.add(keepClassesWithMembersRadioButton);

        JPanel keepOptionPanel = new JPanel(layout);
        keepOptionPanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                                   msg("keepTitle")));

        keepOptionPanel.add(tip(keepClassesAndMembersRadioButton,  "keepTip"),                   constraintsLastStretch);
        keepOptionPanel.add(tip(keepClassMembersRadioButton,       "keepClassMembersTip"),       constraintsLastStretch);
        keepOptionPanel.add(tip(keepClassesWithMembersRadioButton, "keepClassesWithMembersTip"), constraintsLastStretch);

        // Create the also keep panel.
        final JPanel alsoKeepOptionPanel = new JPanel(layout);
        alsoKeepOptionPanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                                       msg("alsoKeepTitle")));

        alsoKeepOptionPanel.add(tip(keepDescriptorClassesCheckBox, "keepDescriptorClassesTip"), constraintsLastStretch);
        alsoKeepOptionPanel.add(tip(keepCodeCheckBox,              "keepCodeTip"),              constraintsLastStretch);

        // Create the allow option panel.
        final JPanel allowOptionPanel = new JPanel(layout);
        allowOptionPanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                                    msg("allowTitle")));

        allowOptionPanel.add(tip(allowShrinkingCheckBox,    "allowShrinkingTip"),    constraintsLastStretch);
        allowOptionPanel.add(tip(allowOptimizationCheckBox, "allowOptimizationTip"), constraintsLastStretch);
        allowOptionPanel.add(tip(allowObfuscationCheckBox,  "allowObfuscationTip"),  constraintsLastStretch);

        conditionDialog = includeKeepSettings ?
            new ClassSpecificationDialog(owner, false, true) :
            null;

        // Create the condition panel.
        final JPanel conditionPanel = new JPanel(layout);
        conditionPanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                                  msg("conditionTitle")));

        final JButton conditionButton = new JButton(msg("edit"));
        conditionButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent actionEvent)
            {
                final ClassSpecification originalCondition =
                    conditionDialog.getClassSpecification();

                int returnValue = conditionDialog.showDialog();
                if (returnValue == APPROVE_OPTION)
                {
                    // Update the condition label.
                    ClassSpecification condition =
                        conditionDialog.getClassSpecification();

                    conditionCommentsField.setText(label(condition.equals(new ClassSpecification()) ? null : condition));
                }
                else
                {
                    // Reset to the original condition.
                    conditionDialog.setClassSpecification(originalCondition);
                }
            }
        });

        // The comments can only be edited in the dialog.
        conditionCommentsField.setEditable(false);

        conditionPanel.add(tip(conditionCommentsField, "commentsTip"),      constraintsStretch);
        conditionPanel.add(tip(conditionButton,        "editConditionTip"), constraintsLast);

        // Create the access panel.
        JPanel accessPanel = new JPanel(layout);
        accessPanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                               msg("access")));

        accessPanel.add(Box.createGlue(),                                labelConstraints);
        accessPanel.add(tip(new JLabel(msg("required")), "requiredTip"), labelConstraints);
        accessPanel.add(tip(new JLabel(msg("not")),      "notTip"),      labelConstraints);
        accessPanel.add(tip(new JLabel(msg("dontCare")), "dontCareTip"), labelConstraints);
        accessPanel.add(Box.createGlue(),                                constraintsLastStretch);

        publicRadioButtons     = addRadioButtonTriplet("Public",     accessPanel);
        finalRadioButtons      = addRadioButtonTriplet("Final",      accessPanel);
        abstractRadioButtons   = addRadioButtonTriplet("Abstract",   accessPanel);
        interfaceRadioButtons  = addRadioButtonTriplet("Interface",  accessPanel);
        annotationRadioButtons = addRadioButtonTriplet("Annotation", accessPanel);
        enumRadioButtons       = addRadioButtonTriplet("Enum",       accessPanel);
        syntheticRadioButtons  = addRadioButtonTriplet("Synthetic",  accessPanel);

        // Create the annotation type panel.
        final JPanel annotationTypePanel = new JPanel(layout);
        annotationTypePanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                                       msg("annotation")));

        annotationTypePanel.add(tip(annotationTypeTextField, "classNameTip"), constraintsLastStretch);

        // Create the class name panel.
        JPanel classNamePanel = new JPanel(layout);
        classNamePanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                                  msg("class")));

        classNamePanel.add(tip(classNameTextField, "classNameTip"), constraintsLastStretch);

        // Create the extends annotation type panel.
        final JPanel extendsAnnotationTypePanel = new JPanel(layout);
        extendsAnnotationTypePanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                                              msg("extendsImplementsAnnotation")));

        extendsAnnotationTypePanel.add(tip(extendsAnnotationTypeTextField, "classNameTip"), constraintsLastStretch);

        // Create the extends class name panel.
        JPanel extendsClassNamePanel = new JPanel(layout);
        extendsClassNamePanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                                         msg("extendsImplementsClass")));

        extendsClassNamePanel.add(tip(extendsClassNameTextField, "classNameTip"), constraintsLastStretch);


        // Create the class member list panel.
        memberSpecificationsPanel = new MemberSpecificationsPanel(this, includeFieldButton);
        memberSpecificationsPanel.setBorder(BorderFactory.createTitledBorder(etchedBorder,
                                                                             msg("classMembers")));

        // Create the Advanced button.
        final JButton advancedButton = new JButton(msg("basic"));
        advancedButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                boolean visible = !alsoKeepOptionPanel.isVisible();

                alsoKeepOptionPanel       .setVisible(visible);
                allowOptionPanel          .setVisible(visible);
                annotationTypePanel       .setVisible(visible);
                extendsAnnotationTypePanel.setVisible(visible);
                conditionPanel            .setVisible(visible);

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
        mainPanel.add(tip(commentsPanel,              "commentsTip"),                    panelConstraints);
        if (includeKeepSettings)
        {
            mainPanel.add(tip(keepOptionPanel,        "keepTitleTip"),                   panelConstraints);
            mainPanel.add(tip(alsoKeepOptionPanel,    "alsoKeepTitleTip"),               panelConstraints);
            mainPanel.add(tip(allowOptionPanel,       "allowTitleTip"),                  panelConstraints);
            mainPanel.add(tip(conditionPanel,         "conditionTip"),                   panelConstraints);
        }
        mainPanel.add(tip(accessPanel,                "accessTip"),                      panelConstraints);
        mainPanel.add(tip(annotationTypePanel,        "annotationTip"),                  panelConstraints);
        mainPanel.add(tip(classNamePanel,             "classTip"),                       panelConstraints);
        mainPanel.add(tip(extendsAnnotationTypePanel, "extendsImplementsAnnotationTip"), panelConstraints);
        mainPanel.add(tip(extendsClassNamePanel,      "extendsImplementsClassTip"),      panelConstraints);
        mainPanel.add(tip(memberSpecificationsPanel,  "classMembersTip"),                stretchPanelConstraints);

        mainPanel.add(tip(advancedButton,             "advancedTip"),                    advancedButtonConstraints);
        mainPanel.add(okButton,                                                          okButtonConstraints);
        mainPanel.add(cancelButton,                                                      cancelButtonConstraints);

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
     * Sets the KeepClassSpecification to be represented in this dialog.
     */
    public void setKeepSpecification(KeepClassSpecification keepClassSpecification)
    {
        boolean            markClasses           = keepClassSpecification.markClasses;
        boolean            markConditionally     = keepClassSpecification.markConditionally;
        boolean            markDescriptorClasses = keepClassSpecification.markDescriptorClasses;
        boolean            markCodeAttributes    = keepClassSpecification.markCodeAttributes;
        boolean            allowShrinking        = keepClassSpecification.allowShrinking;
        boolean            allowOptimization     = keepClassSpecification.allowOptimization;
        boolean            allowObfuscation      = keepClassSpecification.allowObfuscation;
        ClassSpecification condition             = keepClassSpecification.condition;

        // Figure out the proper keep radio button and set it.
        JRadioButton keepOptionRadioButton =
            markConditionally ? keepClassesWithMembersRadioButton :
            markClasses       ? keepClassesAndMembersRadioButton  :
                                keepClassMembersRadioButton;

        keepOptionRadioButton.setSelected(true);

        // Set the other check boxes.
        keepDescriptorClassesCheckBox.setSelected(markDescriptorClasses);
        keepCodeCheckBox             .setSelected(markCodeAttributes);
        allowShrinkingCheckBox       .setSelected(allowShrinking);
        allowOptimizationCheckBox    .setSelected(allowOptimization);
        allowObfuscationCheckBox     .setSelected(allowObfuscation);

        // Set the condition comment and dialog.
        conditionCommentsField.setText(label(condition));
        conditionDialog.setClassSpecification(condition != null ?
                                                  condition :
                                                  new ClassSpecification());

        // Set the rest of the class specification.
        setClassSpecification(keepClassSpecification);
    }


    /**
     * Sets the ClassSpecification to be represented in this dialog.
     */
    public void setClassSpecification(ClassSpecification classSpecification)
    {
        String comments              = classSpecification.comments;
        String annotationType        = classSpecification.annotationType;
        String className             = classSpecification.className;
        String extendsAnnotationType = classSpecification.extendsAnnotationType;
        String extendsClassName      = classSpecification.extendsClassName;
        List   keepFieldOptions      = classSpecification.fieldSpecifications;
        List   keepMethodOptions     = classSpecification.methodSpecifications;

        // Set the comments text area.
        commentsTextArea.setText(comments == null ? "" : comments);

        // Set the access radio buttons.
        setClassSpecificationRadioButtons(classSpecification, ClassConstants.ACC_PUBLIC,      publicRadioButtons);
        setClassSpecificationRadioButtons(classSpecification, ClassConstants.ACC_FINAL,       finalRadioButtons);
        setClassSpecificationRadioButtons(classSpecification, ClassConstants.ACC_ABSTRACT,    abstractRadioButtons);
        setClassSpecificationRadioButtons(classSpecification, ClassConstants.ACC_INTERFACE,   interfaceRadioButtons);
        setClassSpecificationRadioButtons(classSpecification, ClassConstants.ACC_ANNOTATION,  annotationRadioButtons);
        setClassSpecificationRadioButtons(classSpecification, ClassConstants.ACC_ENUM,        enumRadioButtons);
        setClassSpecificationRadioButtons(classSpecification, ClassConstants.ACC_SYNTHETIC,   syntheticRadioButtons);

        // Set the class and annotation text fields.
        annotationTypeTextField       .setText(annotationType        == null ? ""  : ClassUtil.externalType(annotationType));
        classNameTextField            .setText(className             == null ? "*" : ClassUtil.externalClassName(className));
        extendsAnnotationTypeTextField.setText(extendsAnnotationType == null ? ""  : ClassUtil.externalType(extendsAnnotationType));
        extendsClassNameTextField     .setText(extendsClassName      == null ? ""  : ClassUtil.externalClassName(extendsClassName));

        // Set the keep class member option list.
        memberSpecificationsPanel.setMemberSpecifications(keepFieldOptions, keepMethodOptions);
    }


    /**
     * Returns the KeepClassSpecification currently represented in this dialog.
     */
    public KeepClassSpecification getKeepSpecification()
    {
        boolean            markClasses           = !keepClassMembersRadioButton     .isSelected();
        boolean            markConditionally     = keepClassesWithMembersRadioButton.isSelected();
        boolean            markDescriptorClasses = keepDescriptorClassesCheckBox    .isSelected();
        boolean            markCodeAttributes    = keepCodeCheckBox                 .isSelected();
        boolean            allowShrinking        = allowShrinkingCheckBox           .isSelected();
        boolean            allowOptimization     = allowOptimizationCheckBox        .isSelected();
        boolean            allowObfuscation      = allowObfuscationCheckBox         .isSelected();
        ClassSpecification condition             = conditionDialog                  .getClassSpecification();

        return new KeepClassSpecification(markClasses,
                                          markConditionally,
                                          markDescriptorClasses,
                                          markCodeAttributes,
                                          allowShrinking,
                                          allowOptimization,
                                          allowObfuscation,
                                          condition.equals(new ClassSpecification()) ? null : condition,
                                          getClassSpecification());
    }


    /**
     * Returns the ClassSpecification currently represented in this dialog.
     */
    public ClassSpecification getClassSpecification()
    {
        String comments              = commentsTextArea.getText();
        String annotationType        = annotationTypeTextField.getText();
        String className             = classNameTextField.getText();
        String extendsAnnotationType = extendsAnnotationTypeTextField.getText();
        String extendsClassName      = extendsClassNameTextField.getText();

        ClassSpecification classSpecification =
            new ClassSpecification(comments.equals("")              ? null : comments,
                                   0,
                                   0,
                                   annotationType.equals("")        ? null : ClassUtil.internalType(annotationType),
                                   className.equals("") ||
                                   className.equals("*")            ? null : ClassUtil.internalClassName(className),
                                   extendsAnnotationType.equals("") ? null : ClassUtil.internalType(extendsAnnotationType),
                                   extendsClassName.equals("")      ? null : ClassUtil.internalClassName(extendsClassName));

        // Also get the access radio button settings.
        getClassSpecificationRadioButtons(classSpecification, ClassConstants.ACC_PUBLIC,      publicRadioButtons);
        getClassSpecificationRadioButtons(classSpecification, ClassConstants.ACC_FINAL,       finalRadioButtons);
        getClassSpecificationRadioButtons(classSpecification, ClassConstants.ACC_ABSTRACT,    abstractRadioButtons);
        getClassSpecificationRadioButtons(classSpecification, ClassConstants.ACC_INTERFACE,   interfaceRadioButtons);
        getClassSpecificationRadioButtons(classSpecification, ClassConstants.ACC_ANNOTATION, annotationRadioButtons);
        getClassSpecificationRadioButtons(classSpecification, ClassConstants.ACC_ENUM,        enumRadioButtons);
        getClassSpecificationRadioButtons(classSpecification, ClassConstants.ACC_SYNTHETIC,   syntheticRadioButtons);

        // Get the keep class member option lists.
        classSpecification.fieldSpecifications  = memberSpecificationsPanel.getMemberSpecifications(true);
        classSpecification.methodSpecifications = memberSpecificationsPanel.getMemberSpecifications(false);

        return classSpecification;
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
     * Returns a suitable label summarizing the given class specification.
     */
    public String label(ClassSpecification classSpecification)
    {
        return label(classSpecification, -1);
    }


    /**
     * Returns a suitable label summarizing the given class specification at
     * some given index.
     */
    public String label(ClassSpecification classSpecification, int index)
    {
        return
            classSpecification                       == null ? msg("none")                                                                                                        :
            classSpecification.comments              != null ? classSpecification.comments.trim()                                                                                 :
            classSpecification.className             != null ? (msg("class") + ' ' + ClassUtil.externalClassName(classSpecification.className))                                   :
            classSpecification.annotationType        != null ? (msg("classesAnnotatedWith") + ' ' + ClassUtil.externalType(classSpecification.annotationType))                    :
            classSpecification.extendsClassName      != null ? (msg("extensionsOf") + ' ' + ClassUtil.externalClassName(classSpecification.extendsClassName))                     :
            classSpecification.extendsAnnotationType != null ? (msg("extensionsOfClassesAnnotatedWith") + ' ' + ClassUtil.externalType(classSpecification.extendsAnnotationType)) :
            index                                    >= 0    ? (msg("specificationNumber") + index)                                                                               :
                                                               msg("specification");
    }


    /**
     * Sets the appropriate radio button of a given triplet, based on the access
     * flags of the given keep option.
     */
    private void setClassSpecificationRadioButtons(ClassSpecification classSpecification,
                                                   int                flag,
                                                   JRadioButton[]     radioButtons)
    {
        int index = (classSpecification.requiredSetAccessFlags   & flag) != 0 ? 0 :
                    (classSpecification.requiredUnsetAccessFlags & flag) != 0 ? 1 :
                                                                                 2;
        radioButtons[index].setSelected(true);
    }


    /**
     * Updates the access flag of the given keep option, based on the given radio
     * button triplet.
     */
    private void getClassSpecificationRadioButtons(ClassSpecification classSpecification,
                                                   int                flag,
                                                   JRadioButton[]     radioButtons)
    {
        if      (radioButtons[0].isSelected())
        {
            classSpecification.requiredSetAccessFlags   |= flag;
        }
        else if (radioButtons[1].isSelected())
        {
            classSpecification.requiredUnsetAccessFlags |= flag;
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
